#!/usr/bin/env python3
"""Contrastive training pipeline for custom embedding fine-tuning."""

from __future__ import annotations

import gc
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from dotenv import load_dotenv
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import InputExample, SentenceTransformer, losses
from torch.utils.data import DataLoader

load_dotenv()

DEFAULT_TAG = str(os.getenv("PATTERN_EXTRACT_TAG", "default"))
VERSION = str(os.getenv("PATTERN_EXTRACT_VERSION", "v1"))
OUTPUT_ROOT = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", DEFAULT_TAG, VERSION)


@dataclass
class Config:
	output_root: Path = OUTPUT_ROOT

	dataset_path: Path = Path(
		os.getenv(
			"CONTRASTIVE_DATASET_PATH",
			str(OUTPUT_ROOT / "contrastive" / "annotated_code_summaries.csv"),
		)
	)
	text_column: str = os.getenv("CONTRASTIVE_TEXT_COLUMN", "code_summary")
	label_column: str = os.getenv("CONTRASTIVE_LABEL_COLUMN", "label")

	model_name: str = os.getenv("CONTRASTIVE_BASE_MODEL", "BAAI/bge-code-v1")
	hn_base_model: str = os.getenv("CONTRASTIVE_HN_BASE_MODEL", "google-bert/bert-base-uncased")

	loss_margin: float = float(os.getenv("CONTRASTIVE_LOSS_MARGIN", "0.5"))
	use_hard_negatives: bool = os.getenv("CONTRASTIVE_USE_HARD_NEGATIVES", "true").lower() == "true"
	num_hard_negatives: int = int(os.getenv("CONTRASTIVE_NUM_HARD_NEGATIVES", "10"))

	epochs: int = int(os.getenv("CONTRASTIVE_EPOCHS", "3"))
	batch_size: int = int(os.getenv("CONTRASTIVE_BATCH_SIZE", "128"))
	learning_rate: float = float(os.getenv("CONTRASTIVE_LEARNING_RATE", "2e-5"))
	warmup_steps: int = int(os.getenv("CONTRASTIVE_WARMUP_STEPS", "10"))
	max_pairs_per_class: int = int(os.getenv("CONTRASTIVE_MAX_PAIRS_PER_CLASS", "80"))
	max_seq_length: int = int(os.getenv("CONTRASTIVE_MAX_SEQ_LENGTH", "768"))
	min_samples_per_label: int = int(os.getenv("CONTRASTIVE_MIN_SAMPLES_PER_LABEL", "5"))
	seed: int = int(os.getenv("CONTRASTIVE_SEED", "42"))

	save_model_locally: bool = os.getenv("CONTRASTIVE_SAVE_LOCAL", "true").lower() == "true"
	local_model_dir: Path = Path(OUTPUT_ROOT, "contrastive", "saved_models")
	save_run_metadata: bool = os.getenv("CONTRASTIVE_SAVE_METADATA", "true").lower() == "true"
	run_metadata_path: Path = Path(OUTPUT_ROOT, "contrastive", "training_metadata.json")

	upload_to_hub: bool = os.getenv("CONTRASTIVE_UPLOAD_TO_HUB", "false").lower() == "true"
	hub_model_id: str = os.getenv("CONTRASTIVE_HUB_MODEL_ID", "")
	hf_token: str = os.getenv("HF_TOKEN", "")


def ensure_dir(path: Path) -> Path:
	path.mkdir(parents=True, exist_ok=True)
	return path


def set_seed(seed: int) -> None:
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	if torch.cuda.is_available():
		torch.cuda.manual_seed_all(seed)


def clear_memory() -> None:
	gc.collect()
	if torch.cuda.is_available():
		torch.cuda.empty_cache()


def resolve_dataset_path(config: Config) -> Path:
	if config.dataset_path.exists():
		return config.dataset_path

	fallback_candidates = [
		config.output_root / "contrastive" / "annotated_code_summaries.csv",
		config.output_root / "contrastive" / "annotated_contrastive_data.csv",
	]
	for candidate in fallback_candidates:
		if candidate.exists():
			return candidate

	raise FileNotFoundError(
		"Contrastive dataset not found. Expected CSV at: "
		f"{config.dataset_path} or {fallback_candidates}"
	)


def normalize_columns(df: pd.DataFrame, config: Config) -> pd.DataFrame:
	label_col = config.label_column
	text_col = config.text_column

	if label_col not in df.columns and "pattern" in df.columns:
		label_col = "pattern"
	if text_col not in df.columns and "summary" in df.columns:
		text_col = "summary"

	if label_col not in df.columns or text_col not in df.columns:
		raise ValueError(
			f"Dataset must contain label and text columns. "
			f"Configured label='{config.label_column}', text='{config.text_column}'. "
			f"Available columns: {list(df.columns)}"
		)

	normalized = df.copy()
	normalized["__label__"] = normalized[label_col].astype(str).str.strip()
	normalized["__text__"] = normalized[text_col].astype(str).str.strip()
	normalized = normalized[(normalized["__label__"] != "") & (normalized["__text__"] != "")].copy()
	return normalized


def load_and_preprocess_data(config: Config) -> Tuple[pd.DataFrame, LabelEncoder]:
	dataset_path = resolve_dataset_path(config)
	df = pd.read_csv(dataset_path)
	df = normalize_columns(df, config)

	valid_labels = df["__label__"].value_counts()
	valid_labels = valid_labels[valid_labels >= config.min_samples_per_label].index
	df = df[df["__label__"].isin(valid_labels)].reset_index(drop=True)

	if df.empty:
		raise ValueError(
			"No rows left after label filtering. "
			"Lower CONTRASTIVE_MIN_SAMPLES_PER_LABEL or annotate more data."
		)

	encoder = LabelEncoder()
	df["label_enc"] = encoder.fit_transform(df["__label__"])
	df["label"] = df["__label__"]
	df["code_summary"] = df["__text__"]
	return df, encoder


def encode_in_batches(model: SentenceTransformer, texts: Sequence[str], batch_size: int) -> np.ndarray:
	vectors = model.encode(
		list(texts),
		batch_size=batch_size,
		show_progress_bar=True,
		convert_to_numpy=True,
		normalize_embeddings=True,
	)
	return vectors.astype(np.float32)


def mine_hard_negatives(
	texts: Sequence[str],
	labels: Sequence[int],
	base_model_name: str,
	num_hard_negatives: int,
) -> Dict[int, List[int]]:
	print(f"Mining hard negatives with {base_model_name}...")
	base_model = SentenceTransformer(base_model_name, trust_remote_code=True)
	base_embeddings = encode_in_batches(base_model, texts, batch_size=64)
	del base_model
	clear_memory()

	label_arr = np.asarray(labels)
	similarity = base_embeddings @ base_embeddings.T
	hard_map: Dict[int, List[int]] = {}

	for idx in range(len(texts)):
		diff_class_idx = np.where(label_arr != label_arr[idx])[0]
		if len(diff_class_idx) == 0:
			hard_map[idx] = []
			continue
		sims = similarity[idx, diff_class_idx]
		k = min(num_hard_negatives, len(diff_class_idx))
		top_local = np.argsort(sims)[-k:][::-1]
		hard_map[idx] = diff_class_idx[top_local].tolist()

	return hard_map


def build_contrastive_examples(
	texts: Sequence[str],
	labels: Sequence[int],
	max_pairs_per_class: int,
	use_hard_negatives: bool,
	num_hard_negatives: int,
	hn_base_model: str,
) -> List[InputExample]:
	label_to_indices: Dict[int, List[int]] = {}
	for idx, label in enumerate(labels):
		label_to_indices.setdefault(int(label), []).append(idx)

	hard_map: Dict[int, List[int]] = {}
	if use_hard_negatives:
		hard_map = mine_hard_negatives(texts, labels, hn_base_model, num_hard_negatives)

	examples: List[InputExample] = []

	for _, indices in label_to_indices.items():
		if len(indices) < 2:
			continue

		positive_pairs = [
			(indices[i], indices[j])
			for i in range(len(indices))
			for j in range(i + 1, len(indices))
		]

		if len(positive_pairs) > max_pairs_per_class:
			positive_pairs = random.sample(positive_pairs, max_pairs_per_class)

		for a_idx, p_idx in positive_pairs:
			anchor = str(texts[a_idx])
			positive = str(texts[p_idx])
			examples.append(InputExample(texts=[anchor, positive], label=1.0))

			negative_candidates = hard_map.get(a_idx, []) if use_hard_negatives else []
			if not negative_candidates:
				negative_candidates = [
					i for i in range(len(texts)) if int(labels[i]) != int(labels[a_idx])
				]

			if negative_candidates:
				neg_idx = random.choice(negative_candidates)
				negative = str(texts[neg_idx])
				examples.append(InputExample(texts=[anchor, negative], label=0.0))

	random.shuffle(examples)
	return examples


def train_sentence_transformer(config: Config, train_examples: List[InputExample]) -> SentenceTransformer:
	model = SentenceTransformer(config.model_name, trust_remote_code=True)
	model.max_seq_length = config.max_seq_length

	train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=config.batch_size)
	train_loss = losses.ContrastiveLoss(model=model, margin=config.loss_margin)

	model.fit(
		train_objectives=[(train_dataloader, train_loss)],
		epochs=config.epochs,
		warmup_steps=config.warmup_steps,
		optimizer_params={"lr": config.learning_rate},
		show_progress_bar=True,
	)
	return model


def save_model(config: Config, model: SentenceTransformer) -> str:
	ensure_dir(config.local_model_dir)
	model_path = config.local_model_dir / (
		f"full-dataset-contrastive-hn{config.num_hard_negatives if config.use_hard_negatives else 0}"
		f"-ep{config.epochs}"
	)
	model.save(str(model_path))
	print(f"Model saved locally: {model_path}")
	return str(model_path)


def save_metadata(config: Config, dataset: pd.DataFrame, model_path: Optional[str]) -> None:
	if not config.save_run_metadata:
		return

	ensure_dir(config.run_metadata_path.parent)
	class_counts = dataset["label"].value_counts().to_dict()
	payload = {
		"tag": DEFAULT_TAG,
		"version": VERSION,
		"dataset_path": str(resolve_dataset_path(config)),
		"rows_after_filter": int(len(dataset)),
		"num_classes": int(len(class_counts)),
		"class_counts": class_counts,
		"base_model": config.model_name,
		"hard_negative_model": config.hn_base_model,
		"use_hard_negatives": config.use_hard_negatives,
		"num_hard_negatives": config.num_hard_negatives,
		"epochs": config.epochs,
		"batch_size": config.batch_size,
		"learning_rate": config.learning_rate,
		"saved_model_path": model_path,
		"hub_model_id": config.hub_model_id if config.upload_to_hub else "",
	}
	with config.run_metadata_path.open("w", encoding="utf-8") as handle:
		json.dump(payload, handle, indent=2)
	print(f"Saved training metadata: {config.run_metadata_path}")


def upload_to_huggingface(config: Config, model: SentenceTransformer, model_path: Optional[str]) -> None:
	if not config.upload_to_hub:
		return
	if not config.hub_model_id:
		raise ValueError("CONTRASTIVE_HUB_MODEL_ID is required when CONTRASTIVE_UPLOAD_TO_HUB=true")

	from huggingface_hub import HfApi, login

	if config.hf_token:
		login(token=config.hf_token)

	api = HfApi()
	api.create_repo(repo_id=config.hub_model_id, repo_type="model", exist_ok=True)

	try:
		model.push_to_hub(config.hub_model_id, commit_message="Upload fine-tuned contrastive model")
	except Exception:
		if model_path is None:
			raise
		api.upload_folder(
			folder_path=model_path,
			repo_id=config.hub_model_id,
			repo_type="model",
			commit_message="Upload fine-tuned contrastive model",
		)

	print(f"Uploaded: https://huggingface.co/{config.hub_model_id}")


def main(config: Config = Config()) -> None:
	set_seed(config.seed)
	print(f"Using base model: {config.model_name}")

	dataset, _ = load_and_preprocess_data(config)
	texts = dataset["code_summary"].astype(str).tolist()
	labels = dataset["label_enc"].astype(int).tolist()

	print(f"Samples after filtering: {len(dataset)}")
	print(f"Classes after filtering: {dataset['label'].nunique()}")

	examples = build_contrastive_examples(
		texts=texts,
		labels=labels,
		max_pairs_per_class=config.max_pairs_per_class,
		use_hard_negatives=config.use_hard_negatives,
		num_hard_negatives=config.num_hard_negatives,
		hn_base_model=config.hn_base_model,
	)
	if not examples:
		raise ValueError("No training examples generated. Annotate more data per class.")

	print(f"Training examples: {len(examples)}")
	final_model = train_sentence_transformer(config, examples)

	model_path: Optional[str] = None
	if config.save_model_locally:
		model_path = save_model(config, final_model)

	save_metadata(config, dataset, model_path)
	upload_to_huggingface(config, final_model, model_path)

	print("Contrastive training completed.")


if __name__ == "__main__":
	main()
