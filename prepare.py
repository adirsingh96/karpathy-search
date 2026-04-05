"""
prepare.py — One-time setup for btter_bm25.

Run this once before anything else:
    python prepare.py

It will:
  1. Create a .venv with all dependencies
  2. Download the NFCorpus dataset (small: ~3.6k docs, 323 queries)
  3. Save corpus / queries / qrels as JSON in data/
  4. Run a baseline experiment and print the starting NDCG@10
"""

import json
import subprocess
import sys
from pathlib import Path

ROOT     = Path(__file__).parent
VENV_DIR = ROOT / ".venv"
VENV_PY  = VENV_DIR / "bin" / "python"
VENV_PIP = VENV_DIR / "bin" / "pip"
DATA_DIR = ROOT / "data"


def run(cmd, **kwargs):
    subprocess.run(cmd, check=True, **kwargs)


# ── Step 0: ensure we're inside the venv ─────────────────────────────────────
def in_venv() -> bool:
    return sys.prefix != sys.base_prefix

if not in_venv():
    if not VENV_DIR.exists():
        print("[1/4] Creating virtual environment...")
        run([sys.executable, "-m", "venv", str(VENV_DIR)])
    print(f"\nVenv created. Now run:\n")
    print(f"  source {VENV_DIR}/bin/activate")
    print(f"  python prepare.py\n")
    sys.exit(0)


# ── Step 1: install dependencies ──────────────────────────────────────────────
print("[1/4] Installing dependencies...")
run([str(VENV_PIP), "install", "--quiet", "--upgrade", "pip"])
packages = [
    "datasets>=2.0.0",    # HuggingFace datasets (NFCorpus)
    "nltk>=3.8",          # tokenization, stemming
    "tqdm>=4.0",
    "claude-code-sdk",    # autonomous loop (CLI mode)
    "anthropic>=0.7.0",   # Anthropic API (for --use-api mode)
    "python-dotenv>=1.0.0",  # load .env file
]
run([str(VENV_PIP), "install", "--quiet"] + packages)
print("    dependencies installed.")


# ── Step 2: download NLTK data ────────────────────────────────────────────────
print("[2/4] Downloading NLTK data...")
import nltk  # noqa: E402
for pkg in ["punkt", "stopwords"]:
    nltk.download(pkg, quiet=True)
print("    NLTK data ready.")


# ── Step 3: download NFCorpus from HuggingFace ────────────────────────────────
print("[3/4] Downloading NFCorpus dataset...")
DATA_DIR.mkdir(exist_ok=True)

corpus_file  = DATA_DIR / "corpus.json"
queries_file = DATA_DIR / "queries.json"
qrels_file   = DATA_DIR / "qrels.json"

if not (corpus_file.exists() and queries_file.exists() and qrels_file.exists()):
    from datasets import load_dataset  # noqa: E402

    # Corpus
    print("    loading corpus...")
    corpus_ds = load_dataset("BeIR/nfcorpus", "corpus", split="corpus", trust_remote_code=True)
    corpus = {
        row["_id"]: {"title": row.get("title", ""), "text": row.get("text", "")}
        for row in corpus_ds
    }
    with open(corpus_file, "w") as f:
        json.dump(corpus, f)
    print(f"    corpus: {len(corpus):,} documents")

    # Queries (test split)
    print("    loading queries...")
    queries_ds = load_dataset("BeIR/nfcorpus", "queries", split="queries", trust_remote_code=True)
    queries = {row["_id"]: row["text"] for row in queries_ds}
    with open(queries_file, "w") as f:
        json.dump(queries, f)
    print(f"    queries: {len(queries):,} total")

    # Qrels (relevance judgments — test split only)
    print("    loading qrels...")
    qrels_ds = load_dataset("BeIR/nfcorpus-qrels", split="test", trust_remote_code=True)
    qrels: dict[str, dict[str, int]] = {}
    for row in qrels_ds:
        qid  = row["query-id"]
        did  = row["corpus-id"]
        rel  = int(row["score"])
        if qid not in qrels:
            qrels[qid] = {}
        qrels[qid][did] = rel
    with open(qrels_file, "w") as f:
        json.dump(qrels, f)
    print(f"    qrels: {len(qrels):,} queries with judgments")
else:
    print("    dataset already downloaded.")

# ── Step 4: run baseline experiment ──────────────────────────────────────────
print("[4/4] Running baseline experiment...")

# Initialize results.tsv
results_file = ROOT / "results.tsv"
if not results_file.exists() or results_file.stat().st_size == 0:
    with open(results_file, "w") as f:
        f.write("timestamp\trun_id\tndcg10\tmap10\trecall100\tn_queries\teval_time_s\tnotes\n")

run([str(VENV_PY), "search.py"], cwd=ROOT)
run([str(VENV_PY), "eval.py"],   cwd=ROOT)

print("\n✅ Setup complete!")
print()
print("Next steps:")
print("  source .venv/bin/activate")
print("  bash run_experiment.sh        # run one experiment manually")
print("  python loop.py --max-experiments 20  # start the autonomous loop")
