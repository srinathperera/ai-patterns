"""
Self-contained pattern extraction pipeline.

Instructions
- Set GOOGLE_API_KEY in the environment (or the script will prompt once).
- Place input PDFs under data/raw/papers/<tag>/.
- Run this script; it cleans PDFs (unless you point to pre-cleaned text),
  chunks text, calls the LLM with retries, caches chunk outputs, and persists
  per-file and aggregated pattern JSON under outputs/<tag>/<run>/.
"""

from __future__ import annotations

import json
import logging
import os
import time
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Iterable, List, Optional

from langchain.chat_models import init_chat_model
from langchain_core.prompts import PromptTemplate
from pypdf import PdfReader


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# -------------------------- Constants --------------------------
DEFAULT_TAG = "all v2"
DEFAULT_MAX_RETRIES = int(os.getenv("PATTERN_EXTRACT_RETRIES", "3"))
DEFAULT_RETRY_DELAY = float(os.getenv("PATTERN_EXTRACT_RETRY_DELAY", "2.0"))
GEMINI_MAX_CHARS = int(os.getenv("GEMINI_MAX_CHARS", "24000"))
CHUNK_SIZE = GEMINI_MAX_CHARS
CHUNK_OVERLAP = 800


# -------------------------- Prompts (copied from pipeline) --------------------------
PROMPT_OPTIMIZED = """
You are an AI design pattern mining expert.

Extract all **true AI design patterns** mentioned in the following research text. Ignore general software engineering, DevOps, or data engineering patterns.

For each pattern, include:
- Pattern Name :str
- Problem :str
- Context :str
- Solution :str
- Result :str
- Related Patterns :str
- Category :str
- Uses: str
- Thinking: Explain briefly how you identified this as an AI design pattern from the text.

Categories field must be one of the following: 
1. Classical AI
2. Generative AI
3. Agentic AI
4. Prompt Design
5. MLOps (only if specific to ML workflows, not general deployment)
6. AI–Human Interaction
7. LLM-specific
8. Tools Integration
9. Knowledge & Reasoning
10. Planning
11. Personalization

Return only a JSON array. Do not include markdown, extra text, or commentary.

Text:
{text}
"""

PROMPT_RETRY = """\
following is a list of patterns and thinking on how it was extracted in JSON format and paper text from which those patterns were extracted. 
Look for any patterns that are not identified from the paper. If there are any missing design patterns from the paper text, extract them as well and add to the below json array.
if there is any issue with bellow json format, correct it and return only the json array.
""" + PROMPT_OPTIMIZED + """

Extracted patterns so far:
{extracted_patterns}
"""

PROMPT_SUMMARY = """
You are an expert in AI design patterns. 
Your task is to combine the following AI design patterns into a single, unified pattern. 
Use information from all patterns to produce one coherent pattern that includes:

- Pattern Name :str
- Problem :str
- Context :str
- Solution :str
- Result :str
- Related Patterns :str
- Category :str
- Uses: str

Return strictly as JSON. Do not add extra text, explanations, or formatting.

Patterns to combine:
{patterns_text}
"""

PROMPT_MERGE = """
Combine all the following JSON arrays of AI patterns into one deduplicated, coherent JSON array.
If multiple patterns describe similar problems or solutions, merge them carefully.
Return only the final JSON array.

All extracted pattern lists:
{partial_jsons}
"""


# -------------------------- Config --------------------------
@dataclass
class Config:
	tag: str = DEFAULT_TAG
	run_output: Optional[str] = None
	use_cleaned_inputs: bool = False

	def __post_init__(self) -> None:
		root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
		output_base = os.path.join(root, "outputs")
		self.run_output = self.run_output or datetime.now().strftime("%Y%m%d_%H%M%S - Run ")
		self.paper_folder = os.path.join(root, "data/raw/papers", self.tag)
		self.cleaned_folder = os.path.join(root, "data/cleaned/papers", self.tag)
		self.patterns_folder = os.path.join(output_base, self.tag, self.run_output, "extracted_patterns")
		self.patterns_file = os.path.join(self.patterns_folder, "l2_patterns_v2.json")
		os.makedirs(self.patterns_folder, exist_ok=True)


# -------------------------- File + text helpers --------------------------
def normalize_whitespace(text: str) -> str:
	return " ".join(text.split())


def read_pdf(path: str) -> str:
	text_content = ""
	with open(path, "rb") as file:
		reader = PdfReader(file)
		for page in reader.pages:
			text_content += page.extract_text() + "\n"
	return text_content


def read_text(path: str) -> str:
	with open(path, "r") as file:
		return file.read()


def write_text(text: str, path: str) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w") as file:
		file.write(text)


def clean_pdf(path: str) -> str:
	raw_text = read_pdf(path)
	return normalize_whitespace(raw_text)


def clean_pdf_to_file(src_path: str, dest_path: str) -> None:
	cleaned_text = clean_pdf(src_path)
	write_text(cleaned_text, dest_path)


def list_paths(folder_path: str, predicate=None) -> List[str]:
	return [
		os.path.join(folder_path, name)
		for name in os.listdir(folder_path)
		if predicate is None or predicate(name)
	]


def list_pdf_files(folder_path: str) -> List[str]:
	return list_paths(folder_path, lambda name: name.lower().endswith(".pdf"))


def list_text_files(folder_path: str) -> List[str]:
	return list_paths(folder_path, lambda name: name.lower().endswith(".txt"))


def clean_all_pdfs(pdf_folder: str, cleaned_folder: str) -> List[str]:
	os.makedirs(cleaned_folder, exist_ok=True)
	cleaned_files: List[str] = []
	for pdf_path in list_pdf_files(pdf_folder):
		base_name = os.path.basename(pdf_path).replace(".pdf", ".txt")
		out_path = os.path.join(cleaned_folder, f"cleaned_{base_name}")
		clean_pdf_to_file(pdf_path, out_path)
		cleaned_files.append(out_path)
	return cleaned_files


# -------------------------- JSON helpers --------------------------
def parse_json_safe(text: str, delimiter: str = "[]"):
	start, end = text.find(delimiter[0]), text.rfind(delimiter[1]) + 1
	if start != -1 and end != -1:
		try:
			return json.loads(text[start:end])
		except Exception:
			return [] if delimiter == "[]" else {}
	return [] if delimiter == "[]" else {}


def write_json(data: Any, path: str) -> None:
	os.makedirs(os.path.dirname(path), exist_ok=True)
	with open(path, "w") as file:
		json.dump(data, file, indent=2)


def read_json(path: str):
	with open(path, "r") as file:
		return json.load(file)


# -------------------------- LLM setup --------------------------
if not os.environ.get("GOOGLE_API_KEY"):
	os.environ["GOOGLE_API_KEY"] = input("Enter API key for Google Gemini: ")

llm = init_chat_model("gemini-2.5-flash", model_provider="google_genai", temperature=0)


# -------------------------- Chunking --------------------------
def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> List[str]:
	if len(text) <= chunk_size:
		return [text]

	chunks: List[str] = []
	step = max(1, chunk_size - overlap)
	start = 0
	while start < len(text):
		end = min(len(text), start + chunk_size)
		chunks.append(text[start:end])
		if end == len(text):
			break
		start += step
	return chunks


def chunk_cache_path(file_path: str, chunk_idx: int, config: Config) -> str:
	base = os.path.basename(file_path).replace("cleaned_", "").replace(".txt", "")
	cache_dir = os.path.join(config.patterns_folder, "chunk_cache", base)
	os.makedirs(cache_dir, exist_ok=True)
	return os.path.join(cache_dir, f"chunk_{chunk_idx}.json")


def pattern_output_path(file_path: str, config: Config) -> str:
	base_name = os.path.basename(file_path).replace("cleaned_", "").replace(".txt", "_patterns.json")
	return os.path.join(config.patterns_folder, base_name)


# -------------------------- Extraction --------------------------
def extract_patterns_from_text(text: str):
	prompt = PromptTemplate(template=PROMPT_OPTIMIZED, input_variables=["text"])
	prompt_retry = PromptTemplate(template=PROMPT_RETRY, input_variables=["text", "extracted_patterns"])

	try:
		first_pass = llm.invoke(prompt.format(text=text))
	except Exception as exc:
		raise RuntimeError(f"Initial extraction failed: {exc}") from exc

	refined = llm.invoke(prompt_retry.format(text=text, extracted_patterns=first_pass.content))
	return parse_json_safe(refined.content)


def run_chunk_with_retry(chunk_text: str, max_retries: int = DEFAULT_MAX_RETRIES, retry_delay: float = DEFAULT_RETRY_DELAY):
	last_error = None
	for attempt in range(1, max_retries + 1):
		try:
			return extract_patterns_from_text(chunk_text)
		except Exception as exc:
			last_error = exc
			logger.warning("LLM extraction failed on attempt %d/%d: %s", attempt, max_retries, exc)
			if attempt < max_retries:
				time.sleep(retry_delay)
	raise RuntimeError(f"LLM extraction failed after {max_retries} attempts") from last_error


# -------------------------- Orchestration --------------------------
def load_inputs(config: Config) -> List[str]:
	if config.use_cleaned_inputs:
		return list_text_files(config.cleaned_folder)
	return clean_all_pdfs(config.paper_folder, config.cleaned_folder)


def process_file(file_path: str, config: Config) -> List[Any]:
	text = read_text(file_path)
	chunks = chunk_text(text)

	patterns_for_file: List[Any] = []
	for idx, chunk in enumerate(chunks, start=1):
		cache_path = chunk_cache_path(file_path, idx, config)

		if os.path.exists(cache_path):
			try:
				cached = read_json(cache_path)
				logger.info("Using cached chunk %d/%d for %s", idx, len(chunks), file_path)
				patterns_for_file.extend(cached)
				continue
			except Exception as exc:
				logger.warning("Failed to load cached chunk %s (%s); re-extracting", cache_path, exc)

		patterns_chunk = run_chunk_with_retry(chunk)
		write_json(patterns_chunk, cache_path)
		patterns_for_file.extend(patterns_chunk)

	output_path = pattern_output_path(file_path, config)
	write_json(patterns_for_file, output_path)
	return patterns_for_file


def extract_patterns_with_llm(tag: str = DEFAULT_TAG, run_output: Optional[str] = None, use_cleaned_inputs: bool = False) -> List[Any]:
	config = Config(tag=tag, run_output=run_output, use_cleaned_inputs=use_cleaned_inputs)
	input_files = load_inputs(config)

	all_patterns: List[Any] = []
	for file_path in input_files:
		patterns = process_file(file_path, config)
		all_patterns.extend(patterns)
		write_json(all_patterns, config.patterns_file)
	return all_patterns


if __name__ == "__main__":
	extract_patterns_with_llm()
