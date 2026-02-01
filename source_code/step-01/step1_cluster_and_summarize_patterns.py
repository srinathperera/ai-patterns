"""
Self-contained embedding + clustering step.

Instructions
- Set GOOGLE_API_KEY in the environment (or you will be prompted once).
- Ensure patterns JSON exists under outputs/<tag>/<run>/extracted_patterns/l2_patterns_v2.json.
- Run this script to generate embeddings, persist them, UMAP-reduce, DBSCAN-cluster, and
  save clustered data under the same run folder.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Sequence

import numpy as np
import pandas as pd
import umap
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler


# -------------------------- Logging --------------------------
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# -------------------------- Constants --------------------------
DEFAULT_TAG = "all v2"
EMBED_MODEL = "models/gemini-embedding-001"
EMBED_DIM = 768
UMAP_N_NEIGHBORS = 7
UMAP_COMPONENTS = 20
UMAP_METRIC = "cosine"
DBSCAN_EPS = 0.00006479
DBSCAN_MIN_SAMPLES = 1
DBSCAN_METRIC = "cosine"
CLUSTER_OUTPUT_NAME = "umap_clustered_dataset.csv"


# -------------------------- Config --------------------------
@dataclass
class Config:
    tag: str = DEFAULT_TAG
    run_output: str | None = None

    def __post_init__(self) -> None:
        root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
        output_base = os.path.join(root, "outputs")
        self.run_output = self.run_output or datetime.now().strftime("%Y%m%d_%H%M%S - Run ")
        self.patterns_folder = os.path.join(output_base, self.tag, self.run_output, "extracted_patterns")
        self.patterns_file = os.path.join(self.patterns_folder, "l2_patterns_v2.json")
        self.embeddings_file = os.path.join(self.patterns_folder, "pattern_embeddings.csv")
        self.cluster_output = os.path.join(self.patterns_folder, CLUSTER_OUTPUT_NAME)
        os.makedirs(self.patterns_folder, exist_ok=True)


# -------------------------- File helpers --------------------------
def read_json(path: str) -> Any:
    with open(path, "r") as file:
        return json.load(file)


def write_csv(df: pd.DataFrame, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


# -------------------------- Embedding helpers --------------------------
def ensure_api_key() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = input("Enter API key for Google Gemini: ")


def build_embedding_model() -> GoogleGenerativeAIEmbeddings:
    return GoogleGenerativeAIEmbeddings(model=EMBED_MODEL, task_type="clustering")


def combine_pattern_text(pattern: Dict[str, Any]) -> str:
    parts = [
        f"<PatternName>{pattern.get('Pattern Name', 'Unnamed Pattern')}</PatternName>",
        f"<Problem>{pattern.get('Problem', '')}</Problem>",
        f"<Context>{pattern.get('Context', '')}</Context>",
        f"<Solution>{pattern.get('Solution', '')}</Solution>",
        f"<Result>{pattern.get('Result', '')}</Result>",
        f"<Uses>{pattern.get('Uses', '')}</Uses>",
    ]
    return "\n".join(parts)


def generate_embeddings(patterns: Sequence[Dict[str, Any]]) -> pd.DataFrame:
    ensure_api_key()
    model = build_embedding_model()
    combined_texts = [combine_pattern_text(pat) for pat in patterns]
    embeddings = model.embed_documents(combined_texts, task_type="CLUSTERING", output_dimensionality=EMBED_DIM)

    emb_df = pd.DataFrame(embeddings)
    emb_df["Pattern Name"] = [pat.get("Pattern Name", "Unnamed Pattern") for pat in patterns]
    emb_df["Problem"] = [pat.get("Problem", "") for pat in patterns]
    emb_df["Context"] = [pat.get("Context", "") for pat in patterns]
    emb_df["Solution"] = [pat.get("Solution", "") for pat in patterns]
    emb_df["Result"] = [pat.get("Result", "") for pat in patterns]
    emb_df["Uses"] = [pat.get("Uses", "") for pat in patterns]
    emb_df["Description"] = [pat.get("Description", "") for pat in patterns]
    return emb_df


def load_or_create_embeddings(config: Config) -> pd.DataFrame:
    if os.path.exists(config.embeddings_file):
        logger.info("Loading existing embeddings from %s", config.embeddings_file)
        return pd.read_csv(config.embeddings_file)

    logger.info("Generating embeddings from %s", config.patterns_file)
    patterns = read_json(config.patterns_file)
    embeddings_df = generate_embeddings(patterns)
    write_csv(embeddings_df, config.embeddings_file)
    return embeddings_df


# -------------------------- Clustering helpers --------------------------
def select_feature_columns(df: pd.DataFrame) -> List[str]:
    return [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])]


def cluster_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    feature_cols = select_feature_columns(df)
    scaled = StandardScaler().fit_transform(df[feature_cols])

    reducer = umap.UMAP(n_neighbors=UMAP_N_NEIGHBORS, n_components=UMAP_COMPONENTS, metric=UMAP_METRIC)
    umap_space = reducer.fit_transform(scaled)

    dbscan = DBSCAN(eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric=DBSCAN_METRIC)
    labels = dbscan.fit_predict(umap_space)

    logger.info("Clusters found: %d", len(set(labels.tolist())))
    logger.info("Largest cluster size: %d", int(max(np.bincount(labels))))

    clustered = df.copy()
    clustered["cluster"] = labels
    clustered[[f"umap_{i}" for i in range(umap_space.shape[1])]] = umap_space
    return clustered


# -------------------------- Orchestration --------------------------
def run(tag: str = DEFAULT_TAG, run_output: str | None = None) -> pd.DataFrame:
    config = Config(tag=tag, run_output=run_output)
    embeddings_df = load_or_create_embeddings(config)
    clustered_df = cluster_embeddings(embeddings_df)
    write_csv(clustered_df, config.cluster_output)
    return clustered_df


if __name__ == "__main__":
    run()