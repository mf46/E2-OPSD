"""Exemplar retrieval for exemplar-guided teaching (EGT).

Each rollout draws one solved neighboring problem (never the problem itself)
together with its dataset reference solution. The draw is deterministic in
(seed, update, question, rollout slot).
"""
from __future__ import annotations

import hashlib
import json
import os
from typing import NamedTuple, Optional

import numpy as np

# Instruction suffixes that differ between dataset builds of the same question.
# Matching is done on the "core" question with any of these removed.
_KNOWN_SUFFIXES = (
    "Please reason step by step, and put your final answer within \\boxed{}.",
    "Let's think step by step and output the final answer within \\boxed{}.",
    "Please first think step by step inside <think>...</think> tags, "
    "and then put your final answer within \\boxed{}.",
)

_TRAIN_PREFIX = "Problem: "


class Exemplar(NamedTuple):
    """One retrieved exemplar: a different problem plus its reference solution."""

    problem: str
    solution: str
    rank: int  # 1-based position in the precomputed neighbor list


def normalize_question(text: Optional[str]) -> str:
    """Strip whitespace, the `Problem: ` header and known instruction suffixes."""
    if text is None:
        return ""
    core = str(text).strip()
    for suffix in _KNOWN_SUFFIXES:
        if core.endswith(suffix):
            core = core[: -len(suffix)].strip()
            break
    if core.startswith(_TRAIN_PREFIX):
        core = core[len(_TRAIN_PREFIX):].strip()
    return core


def truncate_chars(text: str, budget: int) -> str:
    """Character-budget truncation with an explicit marker."""
    if budget <= 0 or len(text) <= budget:
        return text
    return text[:budget] + " ...[truncated]"


def _stable_seed(*parts) -> int:
    digest = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).hexdigest()
    return int(digest[:8], 16)


def build_corpus_from_parquet(
    parquet_paths,
    answer_template: str = "{solution}\n\nCorrect final answer: {answer}",
    max_rows: int = 0,
    batch_size: int = 512,
) -> dict[str, list[str]]:
    """Build {normalized problem: [privileged text]} from verl train parquet(s).

    Reads row batches and skips the `prompt` column to keep memory low.
    """
    import pyarrow.parquet as pq

    if isinstance(parquet_paths, (str, os.PathLike)):
        parquet_paths = [parquet_paths]

    corpus: dict[str, list[str]] = {}
    rows_seen = 0
    missing_problem = 0
    for path in parquet_paths:
        parquet_file = pq.ParquetFile(path)
        columns = [c for c in ("extra_info", "reward_model") if c in parquet_file.schema_arrow.names]
        for record_batch in parquet_file.iter_batches(batch_size=batch_size, columns=columns):
            extra_infos = record_batch.column("extra_info").to_pylist() if "extra_info" in columns else []
            reward_models = record_batch.column("reward_model").to_pylist() if "reward_model" in columns else []
            for i, extra_info in enumerate(extra_infos):
                rows_seen += 1
                if max_rows and rows_seen > max_rows:
                    return corpus
                extra_info = extra_info or {}
                problem = extra_info.get("problem") or ""
                solution = (extra_info.get("solution") or "").strip()
                if not problem:
                    missing_problem += 1
                    continue
                if not solution:
                    continue
                reward_model = reward_models[i] if i < len(reward_models) else {}
                answer = str(reward_model.get("ground_truth", "")).strip() if isinstance(reward_model, dict) else ""
                core = normalize_question(problem)
                if not core:
                    continue
                text = answer_template.format(solution=solution, answer=answer) if answer else solution
                bucket = corpus.setdefault(core, [])
                if text not in bucket:
                    bucket.append(text)
    if missing_problem:
        print(
            f"[EGT] warning: {missing_problem} rows lack extra_info.problem and were skipped "
            "while building the reference corpus",
            flush=True)
    return corpus


# ------------------------------------------------------------------ #
class ExemplarRetriever:
    """Retrieve one solved neighboring (problem, solution) exemplar."""

    def __init__(
        self,
        neighbors_path: str,
        corpus: dict[str, list[str]],
        max_solutions_per_question: int = 8,
        seed: int = 42,
    ):
        with open(neighbors_path, encoding="utf-8") as handle:
            raw_neighbors = json.load(handle)

        # Normalize once and exclude duplicate or self references.
        self.neighbors: dict[str, list[str]] = {}
        for key, neighbors in raw_neighbors.items():
            core = normalize_question(key)
            if not core:
                continue
            kept_neighbors: list[str] = []
            for position, neighbor in enumerate(neighbors):
                ncore = normalize_question(neighbor)
                if not ncore or ncore == core or ncore in kept_neighbors:
                    continue
                kept_neighbors.append(ncore)
            if kept_neighbors:
                self.neighbors[core] = kept_neighbors

        self.corpus: dict[str, list[str]] = {}
        for key, solutions in corpus.items():
            core = normalize_question(key)
            kept = [s for s in solutions if isinstance(s, str) and s.strip()]
            if core and kept:
                self.corpus[core] = kept[:max_solutions_per_question]

        self.max_solutions_per_question = max_solutions_per_question
        self.seed = int(seed)

    def corpus_size(self) -> int:
        return len(self.corpus)

    def corpus_solutions(self) -> int:
        return sum(len(v) for v in self.corpus.values())

    def retrieve(self, question: str, global_step: int = 0, group_slot: int = 0) -> Optional[Exemplar]:
        """Return an Exemplar, or None when no neighbor has a reference solution.

        Neighbors are visited in a deterministic per-(step, question, slot)
        shuffle, so the first neighbor with a solution is a uniform draw and
        each rollout slot draws independently.
        """
        core = normalize_question(question)
        neighbors = self.neighbors.get(core)
        if not neighbors:
            return None
        slot = int(group_slot)
        rng = np.random.RandomState(_stable_seed(self.seed, global_step, core, slot))
        order = rng.permutation(len(neighbors))
        for idx in order:
            position = int(idx)
            neighbor = neighbors[position]
            if neighbor == core:  # second guard against self-retrieval
                continue
            solutions = self.corpus.get(neighbor)
            if solutions:
                solution = solutions[int(rng.randint(len(solutions)))]
                return Exemplar(problem=neighbor, solution=solution, rank=position + 1)
        return None
