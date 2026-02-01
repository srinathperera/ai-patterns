"""Generate pattern summaries from curated clusters using Gemini."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any, Dict, List

import pandas as pd
from langchain.chat_models import init_chat_model
from langchain.agents import create_agent
from langchain.tools import tool
from langchain.agents.structured_output import ToolStrategy
from pydantic import BaseModel, Field
from tqdm import tqdm


# -------------------------- Config --------------------------
@dataclass
class Config:
    curated_clusters_path: Path = Path("notebooks/cluster_results/curated_clusters_nov25.json")
    raw_patterns_path: Path = Path("outputs/prompt & rag/20251028_085918 - Run/extracted_patterns/all_patterns.json")
    summaries_output_path: Path = Path("outputs/prompt & rag/20251028_085918 - Run/extracted_patterns/pattern_summaries_dec_16.json")
    model_name: str = "gemini-2.5-flash"
    temperature: float = 1.0


# -------------------------- LLM setup --------------------------
def ensure_api_key() -> None:
    if not os.environ.get("GOOGLE_API_KEY"):
        os.environ["GOOGLE_API_KEY"] = input("Enter API key for Google Gemini: ")


def build_llm(config: Config):
    ensure_api_key()
    return init_chat_model(config.model_name, model_provider="google_genai", temperature=config.temperature)


# -------------------------- Tools & Schemas --------------------------
@tool
def summarize_patterns(patterns: str) -> str:
    """Summarize a list of AI design patterns into a single concise description."""

    return (
        "Generalize these AI patterns in a few sentences. Provide one combined pattern description and suggest a name.\n"
        f"AI Patterns:\n\n{patterns}"
    )


class SummaryGenOutput(BaseModel):
    """Structured response for pattern summarization."""

    pattern_summary: str = Field(..., description="Concise summary of the AI design pattern (<=400 words).")


class SummaryAgent:
    def __init__(self, llm):
        self.agent = create_agent(
            llm,
            system_prompt="You are an AI pattern summarization agent that generates concise summaries for AI design patterns.",
            tools=[summarize_patterns],
            response_format=ToolStrategy(SummaryGenOutput),
        )

    def summarize(self, message: str) -> SummaryGenOutput:
        inputs = {"messages": [{"role": "user", "content": message}]}
        return self.agent.invoke(inputs, config={"recursion_limit": 100})["structured_response"]


# -------------------------- Data helpers --------------------------
def load_json_file(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def load_patterns(path: Path) -> pd.DataFrame:
    return pd.read_json(path)


def save_json_file(data: Any, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=4)


# -------------------------- Pipeline --------------------------
def generate_summaries(config: Config) -> List[Dict[str, Any]]:
    curated_clusters = load_json_file(config.curated_clusters_path)
    raw_patterns = load_patterns(config.raw_patterns_path)

    llm = build_llm(config)
    summary_agent = SummaryAgent(llm)

    summaries: List[Dict[str, Any]] = []
    for cluster in tqdm(curated_clusters, desc="Generating Pattern Summaries", ncols=80):
        patterns_df = raw_patterns[raw_patterns["Pattern Name"].isin(set(cluster["l2_patterns"]))]
        payload = json.dumps(patterns_df.to_dict(orient="records"))

        # Retry if the structured response is missing
        response = summary_agent.summarize(payload)
        while not getattr(response, "pattern_summary", None):
            sleep(2)
            response = summary_agent.summarize(payload)

        summaries.append(
            {
                "cluster_id": cluster.get("cluster_id"),
                "cluster_name": cluster.get("short_name"),
                "pattern_summary": response.pattern_summary,
            }
        )

    return summaries


def main(config: Config = Config()) -> None:
    summaries = generate_summaries(config)
    save_json_file(summaries, config.summaries_output_path)


if __name__ == "__main__":
    main()