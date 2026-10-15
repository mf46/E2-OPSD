#!/usr/bin/env python3
"""Build the neighbor map used by exemplar-guided teaching (EGT).

Embed every training problem with Qwen3-Embedding-8B and keep, for each
problem, its nearest neighbors by cosine similarity, excluding the problem
itself (following XRPO, arXiv:2510.06672). Query-side embeddings use the
model's `query` prompt; document-side embeddings are plain.

Similarity is computed in GPU-resident chunks, so a 60k-problem train set never
materializes a 60k x 60k matrix.

Usage (8xH200 node):
  python scripts/build_similar_questions.py \
      --train-parquet data/verl/deepmath_filtered_level5_7/train.parquet \
      --model Qwen/Qwen3-Embedding-8B \
      --out data/egt/qwen3_1_7b_thinking/neighbors.json \
      --num-neighbors 15 --devices cuda:0,cuda:1,cuda:2,cuda:3,cuda:4,cuda:5,cuda:6,cuda:7
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from src.egt_retrieval import normalize_question  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_problems(parquet_paths: list[str]) -> list[str]:
    import pyarrow.parquet as pq

    seen: dict[str, None] = {}
    for path in parquet_paths:
        table = pq.read_table(path, columns=["prompt", "extra_info"])
        prompts = table.column("prompt").to_pylist()
        extra_infos = table.column("extra_info").to_pylist()
        for prompt, extra_info in zip(prompts, extra_infos):
            problem = (extra_info or {}).get("problem") or ""
            if not problem and prompt:
                last = prompt[-1]
                problem = last.get("content", "") if isinstance(last, dict) else ""
            core = normalize_question(problem)
            if core:
                seen.setdefault(core, None)
    return list(seen.keys())


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--train-parquet", nargs="+", required=True)
    parser.add_argument("--model", required=True, help="local path or HF id of the embedding model")
    parser.add_argument("--out", required=True)
    parser.add_argument("--num-neighbors", type=int, default=15,
                        help="neighbors kept per problem, excluding the problem itself (default: 15)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=1024)
    parser.add_argument("--devices", default="", help="comma-separated device list for multi-GPU encoding")
    parser.add_argument("--chunk", type=int, default=2048, help="query rows per similarity chunk")
    args = parser.parse_args()
    if args.num_neighbors < 1:
        parser.error("--num-neighbors must be positive")
    args.topk = args.num_neighbors

    import torch
    from sentence_transformers import SentenceTransformer

    problems = load_problems(args.train_parquet)
    print(f"[build] unique problems: {len(problems)}", flush=True)

    devices = [d.strip() for d in args.devices.split(",") if d.strip()] or None
    model = SentenceTransformer(args.model, model_kwargs={"torch_dtype": torch.bfloat16})
    model.max_seq_length = args.max_seq_length

    encode_kwargs = dict(batch_size=args.batch_size, show_progress_bar=True,
                         convert_to_numpy=True, normalize_embeddings=True)
    if devices:
        encode_kwargs["device"] = devices

    print("[build] encoding documents ...", flush=True)
    doc_emb = model.encode(problems, **encode_kwargs)
    print("[build] encoding queries (prompt_name=query) ...", flush=True)
    query_emb = model.encode(problems, prompt_name="query", **encode_kwargs)

    doc_t = torch.as_tensor(doc_emb, dtype=torch.float16)
    query_t = torch.as_tensor(query_emb, dtype=torch.float16)
    if torch.cuda.is_available():
        doc_t = doc_t.to("cuda:0")
        query_t = query_t.to("cuda:0")
    print(f"[build] embeddings: {tuple(query_t.shape)}", flush=True)

    # Chunked top-(k+1); the self match is dropped by text comparison below so a
    # duplicated problem string can never become its own neighbor.
    k_plus = min(args.topk + 1, len(problems))
    similar: dict[str, list[str]] = {}
    for start in range(0, len(problems), args.chunk):
        end = min(start + args.chunk, len(problems))
        sim = query_t[start:end] @ doc_t.T
        top = torch.topk(sim.float(), k=k_plus, dim=-1)
        top_idx = top.indices.cpu().numpy()
        for row, global_idx in enumerate(range(start, end)):
            core = problems[global_idx]
            neighbors = []
            for j, idx in enumerate(top_idx[row]):
                cand = problems[int(idx)]
                if cand == core:
                    continue
                neighbors.append(cand)
                if len(neighbors) >= args.topk:
                    break
            if neighbors:
                similar[core] = neighbors
        del sim, top
        print(f"[build] neighbors {end}/{len(problems)}", flush=True)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_suffix(".json.inprogress")
    tmp.write_text(json.dumps(similar, ensure_ascii=False), encoding="utf-8")
    tmp.replace(out_path)

    coverage = len(similar) / max(len(problems), 1)
    manifest = {
        "out_path": str(out_path.resolve()),
        "out_sha256": sha256(out_path),
        "out_size_bytes": out_path.stat().st_size,
        "embedding_model": args.model,
        "train_parquet": [str(Path(p).resolve()) for p in args.train_parquet],
        "unique_problems": len(problems),
        "entries": len(similar),
        "coverage": round(coverage, 4),
        "num_neighbors": args.topk,
        "max_seq_length": args.max_seq_length,
        "mean_neighbors": round(sum(len(v) for v in similar.values()) / max(len(similar), 1), 2),
    }
    manifest_path = out_path.with_name(out_path.stem + "_manifest.json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    if coverage < 0.95:
        print(f"WARNING: coverage {coverage:.2%} < 95%", file=sys.stderr)
    print(f"BUILD OK -> {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
