# autoresearch — btter_bm25

## Goal
Maximize **NDCG@10** on the NFCorpus benchmark by improving the search
algorithm in `search.py`. You are free to rewrite the algorithm entirely.

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
1. Read corpus from        `data/corpus.json`   → `{doc_id: {"title": str, "text": str}}`
2. Read queries from       `data/queries.json`  → `{query_id: query_text}`
3. Write rankings to       `rankings.json`      → `{query_id: [doc_id, doc_id, ...]}`
   (ranked best-first, up to 1000 docs per query)
4. Must complete within the **60s time budget**
5. Only use packages already in `.venv` or installable via pip within the script

## Off-limits files (never touch)
- `eval.py` — computes metrics from rankings.json, writes results.tsv
- `prepare.py` — one-time setup
- `run_experiment.sh` — orchestrates the run
- `data/` — downloaded dataset

## Experiment cycle
1. Read `results.tsv` — find best NDCG@10 and what approach produced it
2. Read `search.py` — understand the current algorithm
3. Form ONE hypothesis about what would improve ranking quality
4. Edit `search.py` — implement the improvement
5. Run: `bash run_experiment.sh`
6. Read last row of `results.tsv` for the new score
7. If improved → `git add search.py && git commit -m "ndcg10: X.XXXX — <why>"`
   If same/worse → `git checkout search.py`

## What you can try (from easy to ambitious)

### BM25 parameter tuning (baseline)
- Tune K1, B, DELTA

### BM25 variants
- **BM25+** — add DELTA floor to prevent zero scores for rare terms
- **BM25L** — normalize TF differently to handle long documents better
- **BM25F** — weight title and body fields separately (title is more important)

### Better preprocessing
- Stemming (Porter, Snowball) — reduces vocabulary, improves recall
- Stopword removal — reduces noise
- Bigrams / trigrams — captures phrases like "machine learning"
- Synonym expansion — broaden queries

### Query expansion
- **Pseudo-relevance feedback (RM3)** — take top-k retrieved docs, mine their
  best terms, re-run the query with those added terms
- **Bo1 / KL divergence** — probabilistic query expansion models

### Smarter scoring
- **TF-IDF variants** — log TF, sublinear TF, pivoted normalization
- **Language model with Dirichlet smoothing** — alternative to BM25 IDF
- **Proximity scoring** — boost docs where query terms appear near each other
- **Position-weighted scoring** — terms in title or early in doc score higher

### Ensemble / re-ranking
- Score with multiple methods, combine scores (linear interpolation)
- Use one algorithm for recall (top-1000), another to re-rank top-100

### Learning to rank (if time allows)
- Extract features per (query, doc) pair, train a simple ranker (LambdaMART,
  pointwise linear model) using cross-validation on the training qrels

## Commit format
```
ndcg10: X.XXXX — <one sentence: what changed and why it helps>
```
e.g. `ndcg10: 0.3412 — BM25F with title weight=3.0 outperforms flat BM25`

## Rules
- One algorithmic change per experiment — isolate what works
- If the new approach needs a new pip package, install it inside search.py:
  ```python
  import subprocess, sys
  subprocess.run([sys.executable, "-m", "pip", "install", "-q", "some-package"])
  ```
- If run_experiment.sh errors or times out, revert search.py and move on
- Never commit eval.py, prepare.py, run_experiment.sh, or data/
