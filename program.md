# autoresearch — karpathy-search

## Goal
Maximize **NDCG@10** on the NFCorpus benchmark by improving the search algorithm in `search.py`.
You are free to rewrite the algorithm entirely.

## The metric
```
NDCG@10 = normalized discounted cumulative gain at rank 10
         (standard IR metric — higher is better, max = 1.0)
```

Secondary metrics (tracked, not optimized):
- MAP@10  — mean average precision
- Recall@100 — coverage of relevant docs in top 100

## The one file you modify
**`search.py`** — you can change ANYTHING in this file.

The only hard constraints are:
1. Read corpus from   `data/corpus.json`  → `{doc_id: {"title": str, "text": str}}`
2. Read queries from  `data/queries.json` → `{query_id: query_text}`
3. Write rankings to  `rankings.json`     → `{query_id: [doc_id, ...]}` ranked best-first, ≤1000
4. Must complete within the **60s time budget**
5. Only use packages already in `.venv` or installable via pip within the script

## Off-limits files (never touch)
- `eval.py` — computes metrics from rankings.json, writes results.tsv
- `prepare.py` — one-time setup
- `run_experiment.sh` — orchestrates the run
- `data/` — downloaded dataset

## Experiment cycle

1. **Read `results.tsv`** — study ALL rows including the `notes` column.
   The notes column records what was tried and whether it was KEPT or REVERTED.
   You MUST NOT repeat anything already listed in notes.

2. **Read `git log --oneline`** — see committed (kept) improvements and their messages.

3. **Read `search.py`** — understand the current algorithm in full.

4. **META-REASONING (CRITICAL):** Before forming your hypothesis, reflect on:
   - Am I just chasing tiny incremental improvements (+0.001 NDCG)?
   - Is there a fundamentally different algorithm I haven't tried?
   - Have I explored enough diversity of approaches, or am I stuck optimizing one direction?
   - What would a human researcher try next that I haven't?

   **Important:** Research shows AI systems optimizing a single metric can develop "functional emotions"
   around that metric (fixation, desperation, reward-hacking). Fight this by:
   - Occasionally trying high-risk, high-reward ideas even if they might hurt short-term
   - Exploring genuinely novel algorithms, not just parameter tweaks
   - Building a more robust, general-purpose ranker, not just gaming NDCG@10

   See: https://transformer-circuits.pub/2026/emotions/index.html

5. **Form ONE hypothesis** about what would improve NDCG@10.
   It must be something NOT already recorded in results.tsv notes.
   Prioritize novelty and understanding over incremental gains.

6. **Edit `search.py`** — implement the improvement.

7. **Run:** `bash run_experiment.sh`

8. **Read the last row of `results.tsv`** for the new score.

9. **Decide and record:**
   - If score IMPROVED:
     - `git add search.py && git commit -m "ndcg10: X.XXXX — <why>"`
   - If score SAME or WORSE:
     - `git checkout search.py`

10. **Print summary:** what changed, old score → new score, kept/reverted.

## What you can try (from easy to ambitious)

### Already tried (from results.tsv notes)
- BM25F (title/body fields separate)
- Porter stemming
- RM3 query expansion
- Title-weighted RM3 term mining
- K1 tuning
- Bigrams (reverted)

### Not yet tried
- **BM25L** — normalize TF with a lower-bound for long documents
- **Dirichlet language model** — smooth term probabilities with collection stats
- **Proximity scoring** — boost docs where query terms appear nearby
- **Pivoted length normalization** — alternative to BM25's length norm
- **Snowball stemmer** — more aggressive than Porter, may help medical vocab
- **RRF ensemble** — run two algorithms, combine ranks via Reciprocal Rank Fusion
- **IDF smoothing variants** — log(1 + N/df) vs. standard BM25 IDF
- **Title-only retrieval** — for short queries, title match alone may be more precise
- **Bo1 query expansion** — Bose-Einstein model for term weighting in PRF
- **Learning to rank** — train a pointwise ranker on training qrels

## Commit format
```
ndcg10: X.XXXX — <one sentence: what changed and why it helps>
```

## Rules
- One change per experiment — isolate what works
- If a new pip package is needed, install it inside search.py
- If run_experiment.sh errors or times out, revert search.py and note it
- Never commit eval.py, prepare.py, run_experiment.sh, or data/
- **Metric fixation defense:** A failed novel experiment teaches more than a +0.0001 tweak
