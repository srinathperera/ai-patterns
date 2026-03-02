"""Preprocess embeddings into separate labeled and embeddings CSV files for training."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TAG = str(os.getenv("PATTERN_EXTRACT_TAG", "ai"))
VERSION = str(os.getenv("PATTERN_EXTRACT_VERSION", "v1"))


@dataclass
class Config:
	root: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", DEFAULT_TAG, VERSION)
	synthetic_embeddings_path: Path = Path(root, "code_summary_embeddings_synthetic.csv")
	communities_embeddings_path: Path = Path(root, "code_summary_embeddings_communities.csv")

	use_synthetic_data: bool = os.getenv("USE_SYNTHETIC_DATA", "true").lower() == "true"

	labeled_data_path: Path = Path(root, "labeled_data.csv")
	embeddings_path: Path = Path(root, "embeddings_data.csv")

	file_column: str = "file"
	target_column: str = "pattern"
	verified_column: str = "verified"


def ensure_parent_dir(path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)


def select_dim_columns(df: pd.DataFrame) -> List[str]:
	dim_cols = [col for col in df.columns if col.startswith("dim_")]
	if not dim_cols:
		raise ValueError("No embedding columns found. Expected columns named dim_1, dim_2, ...")
	return dim_cols


def normalize_target_column(series: pd.Series) -> pd.Series:
	values = series.astype(str).str.strip()
	return values.replace({"": np.nan, "none": np.nan, "None": np.nan, "nan": np.nan})


def load_embeddings_csv(path: Path, config: Config) -> pd.DataFrame:
	if not path.exists():
		raise FileNotFoundError(f"Embeddings file not found: {path}")

	df = pd.read_csv(path)
	if config.file_column not in df.columns:
		raise ValueError(f"Missing '{config.file_column}' column in {path}")

	if config.target_column not in df.columns:
		df[config.target_column] = np.nan
	else:
		df[config.target_column] = normalize_target_column(df[config.target_column])

	dim_cols = select_dim_columns(df)
	keep_cols = [config.file_column, config.target_column, *dim_cols]
	return df[keep_cols].copy()


def combine_embeddings(config: Config) -> pd.DataFrame:
	communities_df = load_embeddings_csv(config.communities_embeddings_path, config)
	frames = [communities_df]

	logger.info("Loaded %d community rows from %s", len(communities_df), config.communities_embeddings_path)

	if config.use_synthetic_data:
		synthetic_df = load_embeddings_csv(config.synthetic_embeddings_path, config)
		logger.info("Loaded %d synthetic rows from %s", len(synthetic_df), config.synthetic_embeddings_path)

		dim_communities = set(select_dim_columns(communities_df))
		dim_synthetic = set(select_dim_columns(synthetic_df))
		if dim_communities != dim_synthetic:
			raise ValueError("Synthetic and communities embeddings have different dimensions/columns.")

		frames.append(synthetic_df)
	else:
		logger.info("Synthetic data loading disabled. Using only community embeddings.")

	combined = pd.concat(frames, ignore_index=True)

	before_drop = len(combined)
	combined = combined.drop_duplicates(subset=[config.file_column], keep="first")
	dropped = before_drop - len(combined)
	if dropped > 0:
		logger.warning("Dropped %d duplicate rows by '%s'.", dropped, config.file_column)

	return combined


def split_and_save_training_files(combined: pd.DataFrame, config: Config) -> None:
	dim_cols = select_dim_columns(combined)

	labeled_df = combined[[config.file_column, config.target_column]].copy()
	labeled_df[config.verified_column] = labeled_df[config.target_column].notna()
	embeddings_df = combined[[config.file_column, *dim_cols]].copy()

	ensure_parent_dir(config.labeled_data_path)
	ensure_parent_dir(config.embeddings_path)

	labeled_df.to_csv(config.labeled_data_path, index=False)
	embeddings_df.to_csv(config.embeddings_path, index=False)

	labeled_count = labeled_df[config.target_column].notna().sum()
	unlabeled_count = labeled_df[config.target_column].isna().sum()

	logger.info("Saved labeled_data_path: %s", config.labeled_data_path)
	logger.info("Saved embeddings_path: %s", config.embeddings_path)
	logger.info("Rows: total=%d labeled=%d unlabeled=%d", len(labeled_df), labeled_count, unlabeled_count)


def main(config: Config = Config()) -> None:
	combined = combine_embeddings(config)
	split_and_save_training_files(combined, config)


if __name__ == "__main__":
	main()
