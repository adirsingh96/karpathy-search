# karpathy-search

Autonomous BM25 search algorithm improvement using the [autoresearch](https://github.com/karpathy/autoresearch) pattern by [@karpathy](https://twitter.com/karpathy). A Claude agent iteratively rewrites the search algorithm — not just tunes parameters — to maximize NDCG@10 on the NFCorpus benchmark.

**Best NDCG@10 achieved: `0.3508` — a +7.2% improvement over the plain BM25 baseline in ~10 experiments.**

---

## Results

| Experiment | Algorithm change | NDCG@10 | Δ |
|---|---|---|---|
| Baseline | Plain BM25 (K1=1.5, B=0.75) | 0.3272 | — |
| 1 | Global IDF across title + body fields | 0.3281 | +0.0009 |
| 2 | RM3-style pseudo-relevance feedback | 0.3278 | +0.0006 |
| 3 | Porter stemming for medical term variants | **0.3443** | +0.0171 |
| 4 | Full RM3 — original query terms accumulate feedback weight | 0.3461 | +0.0018 |
| 5 | Title-weighted term mining in RM3 (title tokens × 3) | 0.3468 | +0.0007 |
| 6 | Lower K1 = 1.2 — single medical term hits weighted more | **0.3484** | +0.0016 |
| — | Bigrams — hurt precision, reverted | — | −0.0079 |
| Best | BM25F + Porter stemming + RM3 + K1=1.2 | **0.3508** | **+7.2%** |

---

## How it works

The system follows the [autoresearch](https://github.com/karpathy/autoresearch) pattern:

1. A Claude agent reads `results.tsv` to understand what's been tried
2. Forms a hypothesis about what algorithmic change would improve NDCG@10
3. **Rewrites `search.py`** — not just parameters, but the full algorithm
4. Runs `run_experiment.sh` which evaluates against NFCorpus
5. Commits if NDCG@10 improved; reverts if it didn't
6. Repeats

The key difference from simple hyperparameter tuning: the agent can implement entirely new retrieval strategies — BM25 variants, query expansion, field weighting, ensemble methods — anything that fits the input/output interface contract.

**Interface contract for `search.py`:**
```
INPUT  : data/corpus.json   → {doc_id: {"title": str, "text": str}}
INPUT  : data/queries.json  → {query_id: query_text}
OUTPUT : rankings.json      → {query_id: [doc_id, ...]}  (ranked best-first, ≤1000)
```

---

## What the agent discovered

### BM25F — separate title and body fields (+biggest gain)
Instead of treating the document as one flat bag of words, score title and body separately then combine. Title matches count **3× more** — titles are hand-written summaries, extremely signal-dense in medical literature.

### Porter Stemming (+0.017 — largest single jump)
Reduces vocabulary by conflating morphological variants:
- `inflammation` / `inflammatory` / `inflamed` → `inflamm`
- `obese` / `obesity` / `obesian` → `obes`

Critical for medical text where the same concept appears in many forms.

### RM3 Query Expansion
After the initial retrieval pass, mine the top-10 docs for their most informative terms and add them back to the query. A query for *"vitamin D deficiency"* expands to also search for *"calcium"*, *"rickets"*, *"bone density"* — terms a medical expert would use but the user didn't.

### Lower K1 (1.5 → 1.2)
Controls how quickly term frequency saturates. Lower = a single occurrence of a rare medical term like *"cholecalciferol"* counts almost as much as 10 occurrences. Correct intuition for medical IR where precise terminology matters more than repetition.

### Bigrams — tried, reverted
Adding word pairs to the index hurt: the vocabulary explosion polluted RM3 expansion with low-quality phrase combinations. Unigrams + stemming already handle term normalization well.

---

## Dataset — NFCorpus

[NFCorpus](https://www.cl.uni-heidelberg.de/statnlpgroup/nfcorpus/) (Nutrition Facts Corpus) from the [BEIR benchmark](https://github.com/beir-cellar/beir):

| Property | Value |
|---|---|
| Domain | Medical / nutrition |
| Documents | 3,633 |
| Test queries | 323 |
| Relevance judgments | Human-annotated |

Small enough to evaluate in ~5 seconds per experiment, yet realistic enough that improvements transfer to production IR systems.

---

## Metric — NDCG@10

Normalized Discounted Cumulative Gain at rank 10. Measures both *what* is in the top 10 results and *where* — relevant docs ranked higher are worth more than relevant docs ranked lower.

```
NDCG@10 = your_DCG / ideal_DCG

1.0 = perfect top-10 in perfect order
0.0 = no relevant docs in top 10
```

---

## Setup

```bash
git clone https://github.com/adirsingh96/karpathy-search
cd karpathy-search

# Creates .venv, installs deps, downloads NFCorpus, runs baseline
python prepare.py

source .venv/bin/activate
```

---

## Usage

### Run one experiment manually
```bash
bash run_experiment.sh
```

### Run the autonomous loop
```bash
python loop.py --max-experiments 20 --model sonnet
# or
python loop.py --max-experiments 20 --model haiku   # faster, higher rate limits
```

The loop picks up from wherever it left off — it reads `results.tsv` and the current `search.py` to understand prior progress.

---

## Project structure

```
karpathy-search/
├── search.py           # The search algorithm — the agent rewrites this
├── eval.py             # Evaluation harness (NDCG@10, MAP@10, Recall@100) — do not modify
├── prepare.py          # One-time setup: venv, dataset download, baseline run
├── loop.py             # Autonomous Claude Code experiment loop
├── run_experiment.sh   # Runs one experiment (search + eval)
├── program.md          # Instructions and ideas for the agent
└── results.tsv         # Experiment log (gitignored)
```

---

## What's next

The agent still has room to try:
- **Dirichlet language models** — alternative to BM25 IDF, better for long documents
- **Proximity scoring** — boost docs where query terms appear near each other
- **RRF ensemble** — combine multiple retrieval signals via Reciprocal Rank Fusion
- **Learning to rank** — train a simple ranker on training qrels

---

## Requirements

- Python 3.10+
- [Claude Code CLI](https://claude.ai/code) installed and logged in (for the autonomous loop)
- No GPU needed — pure Python, runs on any machine

---

## Inspired by

[karpathy/autoresearch](https://github.com/karpathy/autoresearch) — the pattern of using a fixed metric + git as a ledger + an AI agent to autonomously run experiments.
