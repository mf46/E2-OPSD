# Copyright 2026 The E2-OPSD Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Canonical evaluation protocol shared by training and offline reporting.

The evaluator does not instantiate models or Ray workers.  It defines the stable
identity, seed and aggregation rules so that a training-time rollout and an
offline replay cannot silently use different conventions.
"""

from __future__ import annotations

import hashlib
import math
import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping


CANONICAL_BENCHMARKS = ("amc23", "aime24", "aime25", "hmmt25")
CANONICAL_QUESTION_COUNTS = {"amc23": 83, "aime24": 30, "aime25": 30, "hmmt25": 30}
CANONICAL_N = 8
CANONICAL_SAMPLING = {"temperature": 0.6, "top_p": 0.95, "top_k": 20}


def stable_hash(value: Any) -> str:
    """Return a process-independent SHA256 for JSON-compatible values."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_hash(value: Any) -> str:
    return hashlib.sha256(str(value).encode("utf-8")).hexdigest()


def model_content_hash(model_path: str | Path) -> str:
    """Hash model files, independent of the enclosing directory or Hub cache."""
    root = Path(model_path).expanduser()
    if not root.is_dir():
        from huggingface_hub import snapshot_download

        # Evaluation runs after model loading, so use that cached snapshot.
        root = Path(snapshot_download(str(model_path), local_files_only=True))
    weights = sorted(root.glob("*.safetensors")) or sorted(root.glob("pytorch_model*.bin"))
    if not weights:
        weights = sorted(root.glob("model_world_size*_rank_*.pt"))
    if not weights:
        if (root / "huggingface").is_dir():
            return model_content_hash(root / "huggingface")
        raise ValueError(f"no model weights found in {root}")
    files = set(weights)
    for pattern in ("config.json", "generation_config.json", "tokenizer*.json", "special_tokens_map.json",
                    "added_tokens.json", "*.index.json", "*.model", "*.tiktoken", "*.jinja",
                    "merges.txt", "vocab.txt", "*.py"):
        files.update(root.glob(pattern))
    config_dir = root / "huggingface"
    if config_dir.is_dir():
        files.update(path for path in config_dir.rglob("*") if path.is_file())
    entries = []
    for path in sorted(files):
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        entries.append((path.relative_to(root).as_posix(), digest.hexdigest()))
    return stable_hash(entries)


def validate_eval_provenance(rows: list[dict], *, n: int) -> dict[str, Any]:
    """Reject incomplete or mixed artifacts before reporting scores."""
    if not rows:
        raise ValueError("evaluation contains no rows")
    identity = {}
    for field in ("model_hash", "manifest_hash", "base_seed", "temperature", "top_p", "top_k", "do_sample", "max_response_length"):
        if any(row.get(field) is None or str(row[field]).strip() in ("", "legacy", "None") for row in rows):
            raise ValueError(f"missing {field}; regenerate evaluation with complete provenance")
        values = {row[field] for row in rows}
        if len(values) != 1:
            raise ValueError(f"mixed {field} values in one evaluation artifact")
        identity[field] = next(iter(values))
    questions = set()
    for row in rows:
        if not row.get("canonical") or "acc" not in row:
            raise ValueError("evaluation requires canonical rows with acc")
        if not row.get("eval_uid") or "eval_seed" not in row or "sample_idx" not in row:
            raise ValueError("missing evaluation sample identity or seed")
        if not 0 <= int(row["sample_idx"]) < n:
            raise ValueError(f"sample_idx outside 0..{n - 1}")
        expected = derive_eval_seed(int(identity["base_seed"]), str(identity["manifest_hash"]),
                                    row["eval_uid"], int(row["sample_idx"]))
        if int(row["eval_seed"]) != expected:
            raise ValueError(f"seed mismatch for {row['eval_uid']}/{row['sample_idx']}")
        questions.add((str(row["eval_uid"]), str(row.get("data_source", "")),
                       text_hash(row.get("prompt", "")), text_hash(row.get("ground_truth", ""))))
    identity["questions_hash"] = stable_hash(sorted(questions))
    return identity


def canonical_manifest_hash(manifest: Mapping[str, Any] | Iterable[Mapping[str, Any]]) -> str:
    """Hash protocol and ordered sample identities, excluding generated text."""
    if isinstance(manifest, Mapping):
        value = dict(manifest)
    else:
        value = {"rows": [dict(row) for row in manifest]}
    return stable_hash(value)


def derive_eval_seed(base_seed: int, manifest_hash: str, eval_uid: Any, sample_idx: int) -> int:
    """Derive a stable vLLM seed independent of batching or worker order."""
    identity = "\x1f".join((str(int(base_seed)), str(manifest_hash), str(eval_uid), str(int(sample_idx))))
    return int.from_bytes(hashlib.blake2s(identity.encode("utf-8"), digest_size=4).digest(), "big") & 0x7FFFFFFF


def protocol_manifest(*, dataset: str = "amc23+aime24+aime25+hmmt25", n: int = CANONICAL_N,
                      sampling: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Build the protocol-only manifest used before prompts are loaded."""
    return {
        "protocol": "canonical-math-eval-v1",
        "dataset": dataset,
        "benchmarks": list(CANONICAL_BENCHMARKS),
        "n": int(n),
        "sampling": dict(CANONICAL_SAMPLING if sampling is None else sampling),
        "question_order": "dataloader-order",
    }


def _row_uid(row: Mapping[str, Any], index: int, n: int = CANONICAL_N) -> str:
    return str(row.get("eval_uid", row.get("uid", f"question-{index // n}")))


def normalize_rollout_rows(rows: Iterable[Mapping[str, Any]], *,
                           n: int = CANONICAL_N) -> list[dict[str, Any]]:
    """Normalize rollout rows without changing their scores.

    ``n`` only drives the fallback identity for rows that carry neither
    ``eval_uid`` nor ``sample_idx``.
    """
    normalized: list[dict[str, Any]] = []
    for index, original in enumerate(rows):
        row = dict(original)
        uid = _row_uid(row, index, n)
        sample_idx = int(row.get("sample_idx", index % n))
        row["eval_uid"] = uid
        row["sample_idx"] = sample_idx
        row.setdefault("prompt_hash", text_hash(row.get("prompt", "")))
        row.setdefault("response_hash", text_hash(row.get("response", "")))
        row.setdefault("reward", float(row.get("score", row.get("reward", 0.0))))
        row.setdefault("acc", float(row.get("acc", row["reward"])))
        row.setdefault("canonical", False)
        normalized.append(row)
    return normalized


def _pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Unbiased pass@k from n samples with c correct (Chen et al., 2021)."""
    if c <= 0:
        return 0.0
    if n - c < k:
        return 1.0
    return 1.0 - math.comb(n - c, k) / math.comb(n, k)


def aggregate_rollouts(rows: Iterable[Mapping[str, Any]], *, n: int = CANONICAL_N,
                       require_math_benchmarks: bool = True,
                       pass_at_ks: Iterable[int] = ()) -> dict[str, Any]:
    """Aggregate mean@N/pass@N with equal benchmark weighting.

    pass_at_ks optionally adds unbiased pass@k estimates (k < n allowed)
    computed from all n samples per question.

    ``acc`` is the official signal.  Raw reward remains available in rows and
    is never substituted when an ``acc`` field is present.
    """
    pass_at_ks = tuple(int(k) for k in pass_at_ks)
    rows = normalize_rollout_rows(rows, n=n)
    if not rows:
        raise ValueError("canonical eval contains no rollout rows")
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if int(row.get("sample_idx", 0)) >= n or int(row.get("sample_idx", 0)) < 0:
            # A global row index is mapped back to a per-question sample index;
            # such rows are marked noncanonical.
            row["sample_idx"] = int(row.get("sample_idx", 0)) % n
            row["canonical"] = False
        source = str(row.get("data_source", row.get("benchmark", "unknown"))).lower()
        row["data_source"] = source
        groups[str(row["eval_uid"])].append(row)
    invalid = {uid: len(items) for uid, items in groups.items() if len(items) != n}
    if invalid:
        raise ValueError(f"each eval_uid must have exactly {n} responses: {invalid}")
    for uid, items in groups.items():
        sample_indices = sorted(int(item.get("sample_idx", -1)) for item in items)
        if sample_indices != list(range(n)):
            raise ValueError(f"{uid}: sample_idx must be a permutation of 0..{n - 1}, got {sample_indices}")
        if len({item["data_source"] for item in items}) != 1:
            raise ValueError(f"{uid}: responses belong to multiple benchmarks")
    benchmarks = list(CANONICAL_BENCHMARKS) if require_math_benchmarks else sorted({r["data_source"] for r in rows})
    if require_math_benchmarks:
        observed = {r["data_source"] for r in rows}
        missing = set(benchmarks) - observed
        unexpected = observed - set(benchmarks)
        if missing or unexpected:
            raise ValueError(f"canonical math eval requires exactly {benchmarks}; missing={sorted(missing)}, unexpected={sorted(unexpected)}")
    by_benchmark: dict[str, dict[str, Any]] = {}
    for benchmark in benchmarks:
        selected = [items for items in groups.values() if items[0]["data_source"] == benchmark]
        if require_math_benchmarks and len(selected) != CANONICAL_QUESTION_COUNTS[benchmark]:
            raise ValueError(f"{benchmark}: expected {CANONICAL_QUESTION_COUNTS[benchmark]} questions, found {len(selected)}")
        if not selected:
            continue
        means = [sum(float(item.get("acc", item.get("reward", 0.0))) for item in items) / n for items in selected]
        passes = [float(any(float(item.get("acc", item.get("reward", 0.0))) > 0.0 for item in items)) for items in selected]
        by_benchmark[benchmark] = {"questions": len(selected), f"mean@{n}": sum(means) / len(means), f"pass@{n}": sum(passes) / len(passes)}
        for k in pass_at_ks:
            k = int(k)
            if not 1 <= k <= n:
                raise ValueError(f"pass@k requires 1 <= k <= n, got k={k} with n={n}")
            per_q = [
                _pass_at_k_unbiased(n, sum(float(item.get("acc", item.get("reward", 0.0))) > 0.0 for item in items), k)
                for items in selected
            ]
            by_benchmark[benchmark][f"pass@{k}"] = sum(per_q) / len(per_q)
    overall = {
        "benchmarks": len(by_benchmark),
        f"mean@{n}": sum(v[f"mean@{n}"] for v in by_benchmark.values()) / len(by_benchmark),
        f"pass@{n}": sum(v[f"pass@{n}"] for v in by_benchmark.values()) / len(by_benchmark),
    }
    for k in pass_at_ks:
        overall[f"pass@{int(k)}"] = sum(v[f"pass@{int(k)}"] for v in by_benchmark.values()) / len(by_benchmark)
    lengths = [float(r["response_length"]) for r in rows if "response_length" in r]
    truncated = [bool(r["truncated"]) for r in rows if "truncated" in r]
    dynamics = {}
    if lengths:
        dynamics["response_length/mean"] = sum(lengths) / len(lengths)
    if truncated:
        dynamics["truncation_rate"] = sum(truncated) / len(truncated)
    return {"overall": overall, "benchmarks": by_benchmark, "responses": len(rows), "dynamics": dynamics}


def write_manifest(path: str | Path, *, rows: Iterable[Mapping[str, Any]],
                   protocol: Mapping[str, Any] | None = None,
                   base_seed: int | None = None,
                   n: int = CANONICAL_N) -> dict[str, Any]:
    normalized = normalize_rollout_rows(rows, n=n)
    sample_manifest = []
    for row in normalized:
        sample_manifest.append({
            "eval_uid": row["eval_uid"],
            "sample_idx": int(row["sample_idx"]),
            "data_source": str(row.get("data_source", row.get("benchmark", "unknown"))).lower(),
            "prompt_hash": row["prompt_hash"],
            "ground_truth_hash": text_hash(row.get("ground_truth", "")),
        })
    payload = dict(protocol or protocol_manifest())
    payload["seed_manifest_hash"] = canonical_manifest_hash(protocol or protocol_manifest())
    if base_seed is not None:
        # Recorded outside the seed manifest hash on purpose: the hash pins the
        # protocol, the seed selects one stream within it.
        payload["base_seed"] = int(base_seed)
    payload["samples"] = sample_manifest
    payload["manifest_hash"] = canonical_manifest_hash(payload)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return payload
