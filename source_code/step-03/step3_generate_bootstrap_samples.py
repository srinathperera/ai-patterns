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
from sklearn import logger
from tqdm import tqdm
from dotenv import load_dotenv
from langchain_core.globals import set_llm_cache

# Completely disable LangChain's LLM caching
set_llm_cache(None)

load_dotenv()


DEFAULT_TAG = str(os.getenv("PATTERN_EXTRACT_TAG", "default"))
VERSION = str(os.getenv("PATTERN_EXTRACT_VERSION", "v1"))
SAMPLES_PER_PATTERN = 5

# -------------------------- Config --------------------------
@dataclass
class Config:
    output_root: str = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),"outputs", DEFAULT_TAG, VERSION)
    patterns_folder: str = os.path.join(output_root, "extracted_patterns")
    curated_clusters_path: Path = Path(output_root, "curated_patterns.json")
    raw_patterns_path: Path = Path(patterns_folder, "l2_patterns.json")
    summaries_output_path: Path = Path(patterns_folder, "pattern_summaries.json")
    code_samples_output_path: Path = Path(output_root, "code_samples.json")
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

@tool
def define_project_idea(pattern_description: str) -> str:
    """
    Tool: Define project idea based on Pattern Description
    """
    
    return f"""Define a realworld AI application idea to implement This AI pattern. Choose random relevant domain. It must be like actual application.
here is some domains: E-commerce, Healthcare, Finance, Education, Social Media, Travel, Real Estate, Entertainment, Food Delivery, Fitness, News Aggregation, Project Management, Customer Support, Event Planning, Job Recruitment, Online Learning, Personal Finance, Blogging Platform, Music Streaming, Video Sharing, Virtual Events, Remote Work Collaboration.
       Here is a pattern description: \n\n{pattern_description}"""

@tool
def define_project_architecture(project_idea: str) -> str:
    """
    Tool: Define project architecture based on project idea.
    """
    
    return f"Define the architecture for the AI application idea. Use suitable framworks and libraries to implement this AI App.\n Use frameworks libraries if need such as tensorflow, pytorch, jax, scikit-learn, lightgbm, xgboost, catboost, fastai, rapids-cuml, transformers, sentence-transformers, tokenizers, spacy, nltk, gensim, trl, accelerate, vllm, langchain, llama-index, chroma, faiss, weaviate, pinecone, milvus, qdrant, elasticsearch, haystack, pandas, numpy, dask, polars, datasets, pyarrow, langgraph, autogen, crewai, opendevin, dspy, semantic-kernel, langsmith, promptlayer, wandb, trulens, evals, guardrails-ai, pydantic, gradio, streamlit, openai, instructor-embedding, cohere, text2vec, clip, openclip, blip, blip2, lavis, diffusers, torchvision, opencv-python, fastapi, ray, bentoml, onnxruntime, tensorrt, tqdm, rich, loguru, python-dotenv, joblib, networkx, phidata, openaimultiswarm, lcel, memgpt, vectorhub, llmdatahub... Here is project idea:\n\n{project_idea}"

@tool
def generate_project_code(project_architecture: str) -> str:
    """
    Tool: Generate code based on project architecture.
    """
    
    return f"Generated code for project architecture, state the code file for realworld application,there can be multiple files,but i want all files in single code.Use suitable libraries and frameworks if needed.Don't add any comment or doc string.Just actual code only. Here is the project architecture:\n\n{project_architecture}"



class SummaryGenOutput(BaseModel):
    """Structured response for pattern summarization."""

    pattern_summary: str = Field(..., description="Concise summary of the AI design pattern (<=400 words).")


class CodeGenOutput(BaseModel):
    """Result from code generation."""
    code: str = Field(..., description="The code body")


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
    

class CodeGenAgent:
    def __init__(self, llm):
        self.agent = create_agent(
            llm,
            system_prompt="You are an AI code generation agent that generates code for AI design patterns. First define a realworld AI application idea to implement This AI pattern. Then define the architecture(libraries definitions) for the AI application idea. Finally generate code based on project architecture as a single file.",
            tools=[define_project_idea,define_project_architecture,generate_project_code],

            response_format=ToolStrategy(CodeGenOutput),
        )

    def generate(self, message: str) -> CodeGenOutput:
        inputs = {"messages": [{"role": "user", "content": message}]}
        return self.agent.invoke(inputs, config={"recursion_limit": 200})["structured_response"]
    


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
def generate_summaries(config: Config, cached_summaries: List[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    curated_clusters = load_json_file(config.curated_clusters_path)
    raw_patterns = load_patterns(config.raw_patterns_path)

    llm = build_llm(config)
    summary_agent = SummaryAgent(llm)

    summaries: List[Dict[str, Any]] = []
    for cluster in tqdm(curated_clusters, desc="Generating Pattern Summaries", ncols=80):
        if cached_summaries:
            cached = next((s for s in cached_summaries if s["cluster_id"] == cluster["cluster_id"]), None)
            if cached:
                summaries.append(cached)
                continue
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

        save_json_file(summaries, config.summaries_output_path)

    return summaries

def generate_code(summary: str, config: Config):
    llm = build_llm(config)
    code_gen_agent = CodeGenAgent(llm)

    response = code_gen_agent.generate(summary)
    while not getattr(response, "code", None):
        sleep(2)
        response = code_gen_agent.generate(summary)

    return response.code

def generate_code_for_summaries(summaries: List[Dict[str, Any]], config: Config, cached_code_samples: List[Dict[str, Any]] = None, samples_per_pattern: int = 1) -> List[Dict[str, Any]]:
    code_outputs = list(cached_code_samples) if cached_code_samples else []
    by_cluster_id = {entry["cluster_id"]: entry for entry in code_outputs}

    for summary in tqdm(summaries, desc="Generating Code for Summaries", ncols=80):
        cluster_id = summary["cluster_id"]
        cluster_name = summary["cluster_name"]

        cluster_entry = by_cluster_id.get(cluster_id)
        if not cluster_entry:
            cluster_entry = {
                "cluster_id": cluster_id,
                "cluster_name": cluster_name,
                "code_samples": [],
            }
            code_outputs.append(cluster_entry)
            by_cluster_id[cluster_id] = cluster_entry
            save_json_file(code_outputs, config.code_samples_output_path)

        pattern_samples = cluster_entry.get("code_samples", [])

        for idx in range(len(pattern_samples), samples_per_pattern):
            code_output = generate_code(summary["pattern_summary"], config)
            code_obj = {
                "filename": f"{cluster_name}/sample_{idx + 1}.py",
                "code": code_output,
            }
            pattern_samples.append(code_obj)
            cluster_entry["code_samples"] = pattern_samples
            save_json_file(code_outputs, config.code_samples_output_path)

    return code_outputs

def main(config: Config = Config()) -> None:
    if os.path.exists(config.summaries_output_path):
        logger.info("Summaries and code outputs already exist. Skipping generation.")
        cached_summaries = load_json_file(config.summaries_output_path)
        summaries = generate_summaries(config, cached_summaries)
    else:
        summaries = generate_summaries(config)

    if os.path.exists(config.code_samples_output_path):
        logger.info("Code samples already exist. Skipping code generation.")
        cached_code_samples = load_json_file(config.code_samples_output_path)
    else:
        cached_code_samples = None

    generate_code_for_summaries(summaries, config,cached_code_samples,SAMPLES_PER_PATTERN)


if __name__ == "__main__":
    main()