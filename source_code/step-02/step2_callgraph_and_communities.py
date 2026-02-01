"""
Generate a call graph from a codebase, detect communities, and write grouped
functions/classes into per-community files.
"""

from __future__ import annotations

import ast
import logging
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import community as community_louvain
import networkx as nx


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logger = logging.getLogger(__name__)


# -------------------------- Config --------------------------
@dataclass
class Config:
    repo_root: Path = Path("repos/cloned_repos")
    output_root: Path = Path("result/repo_callgraph_clusters")

    def output_dir_for_repo(self, repo_path: Path) -> Path:
        return self.output_root / repo_path.name


# -------------------------- AST Visitors --------------------------
class DefinitionCollector(ast.NodeVisitor):
    """Collect fully-qualified function and class nodes."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.functions: Dict[str, ast.AST] = {}
        self.classes: Dict[str, ast.AST] = {}
        self.current_class: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        class_name = f"{self.filename}:{node.name}"
        self.classes[class_name] = node

        previous = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = previous

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prefix = f"{self.filename}:{self.current_class}.{node.name}" if self.current_class else f"{self.filename}:{node.name}"
        self.functions[prefix] = node
        self.generic_visit(node)


class CallGraphVisitor(ast.NodeVisitor):
    """Collect edges (caller -> callee) from function bodies."""

    def __init__(self, filename: str) -> None:
        self.filename = filename
        self.current_function: str | None = None
        self.current_class: str | None = None
        self.edges: List[Tuple[str, str]] = []

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        previous = self.current_class
        self.current_class = node.name
        self.generic_visit(node)
        self.current_class = previous

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        prefix = f"{self.filename}:{self.current_class}.{node.name}" if self.current_class else f"{self.filename}:{node.name}"
        previous_fn = self.current_function
        self.current_function = prefix
        self.generic_visit(node)
        self.current_function = previous_fn

    def visit_Call(self, node: ast.Call) -> None:
        if self.current_function:
            if isinstance(node.func, ast.Name):
                self.edges.append((self.current_function, node.func.id))
            elif isinstance(node.func, ast.Attribute):
                self.edges.append((self.current_function, node.func.attr))
        self.generic_visit(node)


# -------------------------- IO Helpers --------------------------
def iter_python_files(root: Path) -> Iterable[Path]:
    for path in root.rglob("*.py"):
        if path.is_file():
            yield path


def parse_file(path: Path) -> ast.AST | None:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return ast.parse(handle.read(), filename=str(path))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse %s: %s", path, exc)
        return None


# -------------------------- Core Logic --------------------------
def build_call_graph(repo_root: Path) -> nx.DiGraph:
    graph = nx.DiGraph()
    for py_file in iter_python_files(repo_root):
        tree = parse_file(py_file)
        if tree is None:
            continue
        visitor = CallGraphVisitor(str(py_file))
        visitor.visit(tree)
        graph.add_edges_from(visitor.edges)
    return graph


def collect_definitions(repo_root: Path) -> Tuple[Dict[str, ast.AST], Dict[str, ast.AST]]:
    functions: Dict[str, ast.AST] = {}
    classes: Dict[str, ast.AST] = {}
    for py_file in iter_python_files(repo_root):
        tree = parse_file(py_file)
        if tree is None:
            continue
        collector = DefinitionCollector(str(py_file))
        collector.visit(tree)
        functions.update(collector.functions)
        classes.update(collector.classes)
    return functions, classes


def detect_communities(graph: nx.DiGraph) -> Dict[int, List[str]]:
    partition = community_louvain.best_partition(graph.to_undirected())
    clusters: Dict[int, List[str]] = defaultdict(list)
    for node, cluster_id in partition.items():
        clusters[cluster_id].append(node)
    return clusters


def write_cluster_files(clusters: Dict[int, List[str]], functions: Dict[str, ast.AST], classes: Dict[str, ast.AST], output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    for cluster_id, nodes in clusters.items():
        target = output_dir / f"cluster_{cluster_id}.py"
        written_classes: set[str] = set()

        with target.open("w", encoding="utf-8") as handle:
            handle.write(f"# Cluster {cluster_id}\n\n")

            for node_name in nodes:
                for class_name, class_node in classes.items():
                    if node_name.startswith(class_name + ".") and class_name not in written_classes:
                        try:
                            handle.write(ast.unparse(class_node) + "\n\n")
                            written_classes.add(class_name)
                        except Exception:  # noqa: BLE001
                            handle.write(f"# Could not unparse {class_name}\n\n")

                if node_name in functions:
                    try:
                        handle.write(ast.unparse(functions[node_name]) + "\n\n")
                    except Exception:  # noqa: BLE001
                        handle.write(f"# Could not unparse {node_name}\n\n")

        logger.info("Saved %s with %d entries", target, len(nodes))


# -------------------------- Orchestration --------------------------
def process_repository(repo_path: Path, config: Config = Config()) -> None:
    logger.info("Building call graph for %s", repo_path)
    graph = build_call_graph(repo_path)

    logger.info("Collecting definitions for %s", repo_path)
    functions, classes = collect_definitions(repo_path)

    logger.info("Detecting communities")
    clusters = detect_communities(graph)

    output_dir = config.output_dir_for_repo(repo_path)
    logger.info("Writing cluster files to %s", output_dir)
    write_cluster_files(clusters, functions, classes, output_dir)


if __name__ == "__main__":
    cfg = Config()
    for repo in cfg.repo_root.iterdir():
        if repo.is_dir():
            process_repository(repo, cfg)