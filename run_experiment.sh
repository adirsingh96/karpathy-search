#!/usr/bin/env bash
# run_experiment.sh — run one BM25 experiment (search + eval)
# Usage: bash run_experiment.sh

set -euo pipefail
cd "$(dirname "$0")"

# ── Activate venv ────────────────────────────────────────────────────────────
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    if [[ -f .venv/bin/activate ]]; then
        source .venv/bin/activate
    else
        echo "[ERROR] .venv not found. Run: python prepare.py"
        exit 1
    fi
fi

BUDGET=60   # seconds — BM25 is fast; full eval runs in 5-20s

echo ""
echo "$(tput bold)$(tput setaf 6)══ btter_bm25 — experiment run ══$(tput sgr0)"
echo ""

# ── Time budget enforcement ──────────────────────────────────────────────────
if command -v gtimeout &>/dev/null; then
    TIMEOUT_CMD="gtimeout $BUDGET"
elif command -v timeout &>/dev/null; then
    TIMEOUT_CMD="timeout $BUDGET"
else
    echo "$(tput setaf 3)[WARN]$(tput sgr0) No timeout command found; budget not enforced."
    TIMEOUT_CMD=""
fi

# ── Step 1: Search ───────────────────────────────────────────────────────────
echo "$(tput bold)$(tput setaf 6)══ Step 1 / 2 — Ranking (search.py) ══$(tput sgr0)"
T_SEARCH_START=$(date +%s)

$TIMEOUT_CMD python search.py
EXIT_CODE=$?

T_SEARCH_END=$(date +%s)
SEARCH_TIME=$((T_SEARCH_END - T_SEARCH_START))

if [[ $EXIT_CODE -ne 0 ]]; then
    echo "$(tput setaf 1)[ERROR]$(tput sgr0) search.py failed (exit $EXIT_CODE)"
    exit $EXIT_CODE
fi
echo "$(tput setaf 2)[OK]$(tput sgr0) Search complete in ${SEARCH_TIME}s."

# ── Step 2: Eval ─────────────────────────────────────────────────────────────
echo ""
echo "$(tput bold)$(tput setaf 6)══ Step 2 / 2 — Evaluation (eval.py) ══$(tput sgr0)"
python eval.py

# ── Summary ──────────────────────────────────────────────────────────────────
echo ""
echo "$(tput bold)$(tput setaf 6)══ Experiment complete ══$(tput sgr0)"
echo ""
echo "  Latest results:"
tail -1 results.tsv | column -t -s $'\t' 2>/dev/null || tail -1 results.tsv

echo ""
echo "$(tput bold)Next steps:$(tput sgr0)"
echo "  • If score improved  → git add search.py && git commit -m 'ndcg10: X.XXXX — <what changed>'"
echo "  • If score dropped   → git checkout search.py"
echo "  • View all results   → column -t results.tsv"
echo ""
