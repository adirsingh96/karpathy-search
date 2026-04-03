# autoresearch — btter_bm25

## Goal
Maximize **NDCG@10** on the NFCorpus benchmark by tuning BM25 parameters
and preprocessing strategies in `search.py`.

## The metric
```
NDCG@10 = normalized discounted cumulative gain at rank 10
         (standard IR metric — higher is better, max = 1.0)
```
Secondary metrics (tracked but not optimized):
- MAP@10  — mean average precision
- Recall@100 — how many relevant docs appear in top 100

## The one file you modify
**`search.py`** — only the block between:
```
# ===== BEGIN TUNABLE PARAMETERS =====
# ===== END TUNABLE PARAMETERS =====
```

## Off-limits files (never touch)
- `eval.py` — computes metrics from rankings, writes results.tsv
- `prepare.py` — one-time setup
- `run_experiment.sh` — orchestrates the run
- `data/` — downloaded dataset

## Experiment cycle
1. Read `results.tsv` — find best NDCG@10 and what params produced it
2. Read `search.py` — note current parameter values
3. Form ONE hypothesis — change ONE parameter or small related group
4. Edit `search.py` — only within TUNABLE PARAMETERS
5. Run: `bash run_experiment.sh`
6. Read last row of `results.tsv` for the new score
7. If improved → `git add search.py && git commit -m "ndcg10: X.XXXX — <why>"`
   If same/worse → `git checkout search.py`

## Tunable parameter guide

### BM25 core
- `K1` (float, 0.5–3.0): term frequency saturation. Higher = more weight on repeated terms. Default 1.5
- `B` (float, 0.0–1.0): document length normalization. 0 = no normalization, 1 = full. Default 0.75
- `DELTA` (float, 0.0–1.0): BM25+ floor. 0 = standard BM25. Small positive values can help. Default 0.0

### Preprocessing
- `STEMMING`: "none" | "porter" | "snowball" — reduce words to root form
- `REMOVE_STOPWORDS`: True/False — remove common words (the, a, is...)
- `MIN_TOKEN_LEN`: minimum character length to keep a token
- `LOWERCASE`: True/False — case normalization

### Indexing
- `USE_BIGRAMS`: True/False — add bigrams (word pairs) to index alongside unigrams
- `BIGRAM_WEIGHT`: relative weight of bigrams vs unigrams in scoring

### Query
- `QUERY_EXPANSION`: True/False — pseudo-relevance feedback (re-rank using top-k doc terms)
- `EXPANSION_DOCS`: how many top docs to use for expansion
- `EXPANSION_TERMS`: how many terms to add from those docs

## Commit format
```
ndcg10: X.XXXX — <one sentence: what changed and why it helps>
```
4 decimal places, e.g. `ndcg10: 0.3412 — porter stemming reduces vocabulary, improves recall`

## Rules
- ONE change per experiment. Isolate variables.
- Time budget: 60s (BM25 is fast — full eval should take 5-20s)
- Never commit eval.py, prepare.py, run_experiment.sh, or data/
- If run_experiment.sh errors, revert search.py
