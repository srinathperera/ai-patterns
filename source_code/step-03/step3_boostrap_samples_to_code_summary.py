"""Generate code summaries for code samples using Gemini."""

from __future__ import annotations

import json
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from time import sleep
from typing import Any, Dict, List

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langchain_core.globals import set_llm_cache
from pydantic import BaseModel, Field
from tqdm import tqdm

# Completely disable LangChain's LLM caching
set_llm_cache(None)

load_dotenv()

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DEFAULT_TAG = str(os.getenv("PATTERN_EXTRACT_TAG", "ai"))
VERSION = str(os.getenv("PATTERN_EXTRACT_VERSION", "v1"))


@dataclass
class Config:
	output_root: str = os.path.join(
		os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
		"outputs",
		DEFAULT_TAG,
		VERSION,
	)
	code_samples_input_path: Path = Path(output_root, "code_samples.json")
	code_summaries_output_path: Path = Path(output_root, "code_samples_with_summary_synthetic.json")
	model_name: str = os.getenv("CODE_SUMMARY_MODEL", "gemini-2.5-flash")
	temperature: float = float(os.getenv("CODE_SUMMARY_TEMPERATURE", "0.2"))
	max_quota_retries: int = int(os.getenv("MAX_QUOTA_RETRIES", "50"))
	fallback_retry_seconds: float = float(os.getenv("FALLBACK_RETRY_SECONDS", "20"))
	stop_on_quota_exhaustion: bool = os.getenv("STOP_ON_QUOTA_EXHAUSTION", "true").lower() == "true"


def ensure_api_key() -> None:
	if not os.environ.get("GOOGLE_API_KEY"):
		os.environ["GOOGLE_API_KEY"] = input("Enter API key for Google Gemini: ")


def build_llm(config: Config):
	ensure_api_key()
	return init_chat_model(
		config.model_name,
		model_provider="google_genai",
		temperature=config.temperature,
	)


class CodeSummaryOutput(BaseModel):
	code_summary: str = Field(
		...,
		description="Concise and specific community description in 2-3 sentences based on code patterns.",
	)


class CodeSummaryAgent:
	def __init__(self, llm):
		self.generator = llm.with_structured_output(CodeSummaryOutput)

	def summarize(self, code: str) -> CodeSummaryOutput:
		prompt = (
			"You are an expert code description generator. Given a set of patterns that describe a community, "
			"generate a concise and informative description of the community in 2-3 sentences. "
			"Focus on the code patterns and code what does the community represent. "
			"Avoid generic statements and ensure the description is specific to the patterns provided.\n\n"
			"code:\n"
			f"{code}"
		)
		return self.generator.invoke(prompt)


def load_json_file(path: Path) -> Any:
	with path.open("r", encoding="utf-8") as handle:
		return json.load(handle)


def save_json_file(data: Any, path: Path) -> None:
	path.parent.mkdir(parents=True, exist_ok=True)
	with path.open("w", encoding="utf-8") as handle:
		json.dump(data, handle, indent=4)


def is_quota_error(exc: Exception) -> bool:
	message = str(exc)
	return "RESOURCE_EXHAUSTED" in message or "quota" in message.lower() or "429" in message


def parse_retry_seconds(exc: Exception) -> float | None:
	message = str(exc)
	patterns = [
		r"Please retry in\s+([0-9]+(?:\.[0-9]+)?)s",
		r"'retryDelay':\s*'([0-9]+)s'",
	]
	for pattern in patterns:
		match = re.search(pattern, message)
		if match:
			return float(match.group(1))
	return None


def invoke_with_quota_retry(callable_fn, config: Config, op_name: str):
	for attempt in range(1, config.max_quota_retries + 1):
		try:
			return callable_fn()
		except Exception as exc:
			if not is_quota_error(exc):
				raise

			delay = parse_retry_seconds(exc)
			if delay is None:
				delay = min(config.fallback_retry_seconds * (2 ** min(attempt - 1, 3)), 300)

			logger.warning(
				"%s hit free-tier quota/rate limit (attempt %s/%s). Waiting %.1fs before retry.",
				op_name,
				attempt,
				config.max_quota_retries,
				delay,
			)
			sleep(delay + 1)

	raise RuntimeError(f"{op_name} failed after {config.max_quota_retries} quota retries")


def merge_input_with_cached(
	input_samples: List[Dict[str, Any]],
	cached_samples: List[Dict[str, Any]] | None,
) -> List[Dict[str, Any]]:
	if not cached_samples:
		return input_samples

	cached_map: Dict[tuple[Any, str], str] = {}
	for cluster in cached_samples:
		cluster_id = cluster.get("cluster_id")
		for sample in cluster.get("code_samples", []):
			filename = sample.get("filename")
			summary = sample.get("code_summary")
			if filename and summary:
				cached_map[(cluster_id, filename)] = summary

	merged: List[Dict[str, Any]] = []
	for cluster in input_samples:
		cluster_id = cluster.get("cluster_id")
		merged_cluster = {
			"cluster_id": cluster_id,
			"cluster_name": cluster.get("cluster_name"),
			"code_samples": [],
		}
		for sample in cluster.get("code_samples", []):
			filename = sample.get("filename")
			merged_sample = {
				"filename": filename,
				"code": sample.get("code", ""),
			}
			cached_summary = cached_map.get((cluster_id, filename))
			if cached_summary:
				merged_sample["code_summary"] = cached_summary
			merged_cluster["code_samples"].append(merged_sample)
		merged.append(merged_cluster)

	return merged


def generate_code_summaries(config: Config) -> List[Dict[str, Any]]:
	if not config.code_samples_input_path.exists():
		raise FileNotFoundError(f"Input file not found: {config.code_samples_input_path}")

	input_samples = load_json_file(config.code_samples_input_path)
	cached_output = (
		load_json_file(config.code_summaries_output_path)
		if config.code_summaries_output_path.exists()
		else None
	)
	output_samples = merge_input_with_cached(input_samples, cached_output)

	llm = build_llm(config)
	summary_agent = CodeSummaryAgent(llm)

	for cluster in tqdm(output_samples, desc="Generating Code Summaries", ncols=80):
		for sample in cluster.get("code_samples", []):
			if sample.get("code_summary"):
				continue

			code = sample.get("code", "")
			if not code:
				sample["code_summary"] = ""
				save_json_file(output_samples, config.code_summaries_output_path)
				continue

			try:
				response = invoke_with_quota_retry(
					lambda: summary_agent.summarize(code),
					config,
					"Code summary generation",
				)
			except RuntimeError as exc:
				if config.stop_on_quota_exhaustion:
					logger.warning("Stopping early due to free-tier quota exhaustion: %s", exc)
					save_json_file(output_samples, config.code_summaries_output_path)
					return output_samples
				raise

			sample["code_summary"] = response.code_summary
			save_json_file(output_samples, config.code_summaries_output_path)

	return output_samples


def main(config: Config = Config()) -> None:
	generated = generate_code_summaries(config)
	logger.info("Completed code summary generation for %s clusters.", len(generated))
	logger.info("Saved output to %s", config.code_summaries_output_path)


if __name__ == "__main__":
	main()
