"""
Generate Gemini embeddings for code samples in JSON and export as CSV.
"""

from __future__ import annotations

import logging
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, List, Sequence

import pandas as pd
from dotenv import load_dotenv

load_dotenv()

try:
    import google.generativeai as genai
except ImportError as exc:  # noqa: BLE001
    raise ImportError("Install google-generativeai via `pip install google-generativeai`.") from exc


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)

DEFAULT_TAG = str(os.getenv("PATTERN_EXTRACT_TAG", "default"))
VERSION = str(os.getenv("PATTERN_EXTRACT_VERSION", "v1"))


# -------------------------- Config --------------------------
@dataclass
class Config:
    input_json_path: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", DEFAULT_TAG, VERSION, "code_samples.json")
    # Where to write the embeddings CSV
    output_path: Path = Path(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", DEFAULT_TAG, VERSION, "pattern_embeddings.csv")
    model_name: str = "models/gemini-embedding-001"
    batch_size: int = 250


# -------------------------- API setup --------------------------
def configure_client() -> None:
    api_key = os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("Set GOOGLE_API_KEY or GEMINI_API_KEY before running.")
    genai.configure(api_key=api_key)


# -------------------------- IO helpers --------------------------
def load_code_samples(json_path: Path) -> List[dict[str, Any]]:
    if not json_path.exists():
        raise FileNotFoundError(f"code samples JSON not found: {json_path}")

    payload = json.loads(json_path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError("code_samples.json must contain a top-level list.")

    samples: List[dict[str, Any]] = []
    for cluster in payload:
        cluster_id = cluster.get("cluster_id")
        cluster_name = cluster.get("cluster_name", "")
        code_samples = cluster.get("code_samples", [])

        if not isinstance(code_samples, list):
            logger.warning("Skipping cluster with invalid code_samples list: %s", cluster_name)
            continue

        for sample in code_samples:
            code_text = sample.get("code", "")
            if not code_text:
                continue
            samples.append(
                {
                    "cluster_id": cluster_id,
                    "cluster_name": cluster_name,
                    "filename": sample.get("filename", "unknown.py"),
                    "code": code_text,
                }
            )

    if not samples:
        raise ValueError("No code samples with non-empty code were found in the input JSON.")
    return samples


# -------------------------- Embedding helpers --------------------------
def embed_texts(texts: Sequence[str], model_name: str) -> List[List[float]]:
    embeddings: List[List[float]] = []
    for text in texts:
        response = genai.embed_content(model=model_name, content=text, task_type="SEMANTIC_SIMILARITY")
        embeddings.append(response["embedding"])
    return embeddings


def chunk(sequence: Sequence, size: int) -> Iterable[Sequence]:
    for start in range(0, len(sequence), size):
        yield sequence[start : start + size]


# -------------------------- Main pipeline --------------------------
def build_embeddings_dataframe(config: Config) -> pd.DataFrame:
    samples = load_code_samples(config.input_json_path)
    total_samples = len(samples)
    records: List[dict] = []

    for start_idx in range(0, total_samples, config.batch_size):
        batch = samples[start_idx : start_idx + config.batch_size]
        logger.info(
            "Processing samples %d-%d / %d",
            start_idx + 1,
            min(start_idx + len(batch), total_samples),
            total_samples,
        )

        texts = [f"# File: {sample['filename']}\n{sample['code']}" for sample in batch]
        embeddings = embed_texts(texts, config.model_name)

        for sample, embedding_vector in zip(batch, embeddings):
            records.append(
                {
                    "cluster_name": sample["cluster_name"],
                    "file": sample["filename"],
                    "embedding": embedding_vector,
                }
            )

    if not records:
        raise ValueError("No embeddings were generated; ensure .py files have content.")

    embedding_length = len(records[0]["embedding"])
    rows = []
    for record in records:
        row = {f"dim_{i+1}": value for i, value in enumerate(record["embedding"])}
        row["pattern"] = record["cluster_name"]
        row["file"] = record["file"]
        rows.append(row)

    df = pd.DataFrame(rows)
    logger.info("Generated embeddings for %d files with %d dimensions", len(df), embedding_length)
    return df

 
def save_embeddings(df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    logger.info("Saved embeddings to %s", output_path.resolve())


def main(config: Config = Config()) -> None:
    configure_client()
    embeddings_df = build_embeddings_dataframe(config)
    save_embeddings(embeddings_df, config.output_path)


if __name__ == "__main__":
    main()
