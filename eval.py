"""
eval.py — Evaluation script for btter_bm25.

DO NOT MODIFY — this file is the ground truth evaluator.
Reads rankings.json produced by search.py and computes:
  - NDCG@10  (primary metric to optimize)
  - MAP@10
  - Recall@100
Appends one row to results.tsv.
"""

import json
import math
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent


def ndcg_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """Normalized Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, did in enumerate(ranked_ids[:k]):
        rel = qrels.get(did, 0)
        dcg += rel / math.log2(i + 2)

    ideal_rels = sorted(qrels.values(), reverse=True)[:k]
    idcg = sum(rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels))

    return dcg / idcg if idcg > 0 else 0.0


def ap_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int = 10) -> float:
    """Average Precision at rank k (binary relevance: rel > 0 counts)."""
    hits, total_ap = 0, 0.0
    for i, did in enumerate(ranked_ids[:k]):
        if qrels.get(did, 0) > 0:
            hits += 1
            total_ap += hits / (i + 1)
    n_rel = sum(1 for r in qrels.values() if r > 0)
    return total_ap / min(n_rel, k) if n_rel > 0 else 0.0


def recall_at_k(ranked_ids: list[str], qrels: dict[str, int], k: int = 100) -> float:
    """Recall at rank k."""
    n_rel = sum(1 for r in qrels.values() if r > 0)
    if n_rel == 0:
        return 0.0
    hits = sum(1 for did in ranked_ids[:k] if qrels.get(did, 0) > 0)
    return hits / n_rel


def main():
    t0 = time.time()

    rankings_file = ROOT / "rankings.json"
    qrels_file    = ROOT / "data" / "qrels.json"
    results_file  = ROOT / "results.tsv"

    if not rankings_file.exists():
        print("[eval] ERROR: rankings.json not found — did search.py run?")
        sys.exit(1)

    if not qrels_file.exists():
        print("[eval] ERROR: data/qrels.json not found — run prepare.py")
        sys.exit(1)

    with open(rankings_file) as f:
        rankings: dict[str, list[str]] = json.load(f)

    with open(qrels_file) as f:
        all_qrels: dict[str, dict[str, int]] = json.load(f)  # {qid: {doc_id: relevance}}

    # Evaluate only queries that have qrels
    ndcg_scores, map_scores, recall_scores = [], [], []

    for qid, qrels in all_qrels.items():
        if not qrels:
            continue
        ranked = rankings.get(qid, [])
        ndcg_scores.append(ndcg_at_k(ranked, qrels, k=10))
        map_scores.append(ap_at_k(ranked, qrels, k=10))
        recall_scores.append(recall_at_k(ranked, qrels, k=100))

    ndcg10   = sum(ndcg_scores)   / len(ndcg_scores)   if ndcg_scores else 0.0
    map10    = sum(map_scores)    / len(map_scores)     if map_scores else 0.0
    recall100 = sum(recall_scores) / len(recall_scores) if recall_scores else 0.0

    elapsed = time.time() - t0

    print()
    print("=" * 50)
    print(f"  NDCG@10   : {ndcg10:.4f}   ← OPTIMIZE THIS")
    print(f"  MAP@10    : {map10:.4f}")
    print(f"  Recall@100: {recall100:.4f}")
    print(f"  Queries   : {len(ndcg_scores)}")
    print("=" * 50)

    # Get git commit hash
    try:
        run_id = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        run_id = "unknown"

    # Write header if needed
    if not results_file.exists() or results_file.stat().st_size == 0:
        with open(results_file, "w") as f:
            f.write("timestamp\trun_id\tndcg10\tmap10\trecall100\tn_queries\teval_time_s\tnotes\n")

    timestamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    with open(results_file, "a") as f:
        f.write(f"{timestamp}\t{run_id}\t{ndcg10:.4f}\t{map10:.4f}\t{recall100:.4f}"
                f"\t{len(ndcg_scores)}\t{elapsed:.1f}\t\n")

    print(f"\n[results] Appended to results.tsv")


if __name__ == "__main__":
    main()
