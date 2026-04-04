#!/usr/bin/env python3
"""
loop.py — Autonomous experiment loop for btter_bm25.

Calls the Claude Code CLI as a subprocess so the agent can read files,
edit search.py, run experiments, and commit improvements — fully hands-free.

Requirements:
  - Claude Code CLI installed  →  npm install -g @anthropic-ai/claude-code
  - .venv activated            →  source .venv/bin/activate
  - One-time setup done        →  python prepare.py

Usage:
  python loop.py                        # runs indefinitely
  python loop.py --max-experiments 20   # stop after N experiments
  python loop.py --model sonnet         # smarter model (default: haiku)
"""

import argparse
import shutil
import signal
import subprocess
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).parent

EXPERIMENT_PROMPT = """\
You are the autonomous research agent for btter_bm25. Your job is to run
ONE complete experiment cycle. Follow these steps exactly:

1. Read program.md — understand the rules, the interface contract, and ideas to try.
2. Read results.tsv — find the current best NDCG@10 and what algorithm produced it.
   If the file has only a header row, the baseline is 0.0.
3. Read search.py — understand the current algorithm fully.
4. Form ONE hypothesis about what algorithmic change would improve NDCG@10.
   You can tune parameters OR rewrite the algorithm entirely — both are valid.
   Write your reasoning in one sentence before acting.
5. Edit search.py — implement your improvement.
   The only hard constraints are the interface contract in program.md:
     - Read from data/corpus.json and data/queries.json
     - Write rankings.json as {query_id: [doc_id, ...]} ranked best-first
   Everything else is fair game.
6. Run the experiment (budget: 60s):
       bash run_experiment.sh
7. Read the last data row of results.tsv to get the new NDCG@10 score.
8. Decide:
   - Score IMPROVED  → git add search.py && git commit -m "ndcg10: X.XXXX — <why>"
   - Score SAME or WORSE → git checkout search.py   (revert, no commit)
9. Print a concise summary: what changed, old score → new score, kept/reverted.

Rules:
- One algorithmic change per experiment. Isolate what works.
- Commit message must start with "ndcg10: X.XXXX — " (4 decimal places).
- If you need a new pip package, install it inside search.py with subprocess.
- If run_experiment.sh errors or times out, revert search.py and move on.
- Never commit eval.py, prepare.py, run_experiment.sh, or data/.
"""

EXPERIMENT_TIMEOUT_S = 600   # 10 min ceiling; agent needs ~1-3 min to read/think/edit/run


def find_claude_cli() -> str:
    path = shutil.which("claude")
    if path:
        return path
    raise FileNotFoundError(
        "Claude Code CLI not found.\n"
        "  Install:  npm install -g @anthropic-ai/claude-code\n"
        "  Login:    claude login"
    )


def run_one_experiment(iteration: int, model: str) -> bool:
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"\n{'─' * 60}")
    print(f"  Experiment {iteration}  [{ts}]")
    print(f"{'─' * 60}\n", flush=True)

    try:
        cli = find_claude_cli()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        return False

    cmd = [
        cli,
        "--print",
        "--model", model,
        "--allowed-tools", "Read,Edit,Bash,Glob,Grep",
        "--dangerously-skip-permissions",
        "--output-format", "text",
        EXPERIMENT_PROMPT,
    ]

    try:
        result = subprocess.run(
            cmd,
            cwd=str(ROOT),
            stdout=None,
            stderr=None,
            timeout=EXPERIMENT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        print(f"\n[loop] Timed out after {EXPERIMENT_TIMEOUT_S}s — reverting search.py.")
        subprocess.run(["git", "checkout", "search.py"], cwd=str(ROOT))
        return False

    if result.returncode != 0:
        print(f"\n[loop] claude exited with code {result.returncode}.")
        return False

    return True


def main():
    parser = argparse.ArgumentParser(description="btter_bm25 — autonomous experiment loop")
    parser.add_argument("--max-experiments", type=int, default=0, metavar="N",
                        help="Stop after N experiments (0 = run forever)")
    parser.add_argument("--model", default="haiku",
                        help="Claude model: haiku (default), sonnet, opus")
    args = parser.parse_args()

    if not (ROOT / "search.py").exists():
        print("[ERROR] search.py not found. Run from the btter_bm25 directory.")
        sys.exit(1)

    if not (ROOT / "data" / "corpus.json").exists():
        print("[ERROR] data/ not found. Run: python prepare.py")
        sys.exit(1)

    try:
        find_claude_cli()
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print()
    print("=" * 60)
    print("  btter_bm25 — Autonomous Experiment Loop")
    print(f"  Model      : {args.model}")
    print(f"  Budget     : {'∞' if args.max_experiments == 0 else args.max_experiments} experiments")
    print(f"  Dataset    : NFCorpus (3.6k docs, 323 queries)")
    print(f"  Metric     : NDCG@10")
    print()
    print("  Press Ctrl+C to stop after the current experiment.")
    print("=" * 60)

    stop = False

    def _on_sigint(sig, frame):
        nonlocal stop
        if not stop:
            print("\n\n[loop] Ctrl+C — stopping after this experiment...")
            stop = True
        else:
            sys.exit(1)

    signal.signal(signal.SIGINT, _on_sigint)

    iteration, errors = 1, 0
    while not stop:
        if args.max_experiments > 0 and iteration > args.max_experiments:
            print(f"\n[loop] Reached {args.max_experiments} experiments. Done.")
            break

        success = run_one_experiment(iteration, args.model)
        errors  = 0 if success else errors + 1
        if errors >= 3:
            print("[loop] 3 consecutive errors — exiting.")
            break
        iteration += 1

    # Summary
    print()
    print("=" * 60)
    print("  Loop complete.")
    results_file = ROOT / "results.tsv"
    if results_file.exists():
        rows = [l for l in results_file.read_text().splitlines() if l.strip()]
        if len(rows) > 1:
            best = max(
                (r.split("\t") for r in rows[1:] if len(r.split("\t")) > 2),
                key=lambda r: float(r[2]) if len(r) > 2 else 0,
                default=None,
            )
            if best:
                print(f"  Best NDCG@10 : {best[2]}")
                print(f"  Run          : {best[1]}")
    print()
    print("  git log --oneline       ← to see kept experiments")
    print("  column -t results.tsv   ← to review all scores")
    print("=" * 60)


if __name__ == "__main__":
    main()
