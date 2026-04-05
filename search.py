"""
search.py — Search algorithm for btter_bm25.

THE FILE THE AGENT REWRITES.

Hard interface contract (do not break these):
  INPUT  : data/corpus.json   → {doc_id: {"title": str, "text": str}}
  INPUT  : data/queries.json  → {query_id: query_text}
  OUTPUT : rankings.json      → {query_id: [doc_id, ...]}   (ranked best-first, ≤1000 per query)

Everything else is fair game — rewrite the algorithm however you like.
Current approach: BM25F + pseudo-relevance feedback (RM3-style query expansion).
"""

import subprocess, sys
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "nltk"], check=False)
import nltk
try:
    nltk.data.find("tokenizers/punkt")
except LookupError:
    nltk.download("punkt", quiet=True)

import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_FILE = ROOT / "rankings.json"

# ── Parameters ────────────────────────────────────────────────────────────────

K1           = 0.9    # TF saturation
B_TITLE      = 0.4    # length norm for title field
B_BODY       = 0.75   # length norm for body field
TITLE_WEIGHT = 3.2    # title match multiplier vs body
DELTA        = 0.0    # BM25+ floor

# Proximity scoring
PROX_WEIGHT  = 0.20   # fraction of top BM25F score added as proximity bonus

# Pseudo-relevance feedback (RM3-style)
FB_DOCS      = 10     # number of top docs to mine for expansion terms
FB_TERMS     = 15     # number of expansion terms to add
FB_ALPHA     = 0.5    # weight of original query vs expansion (0=all expansion, 1=no expansion)

# RRF combination
RRF_K        = 60     # RRF rank offset (standard: 60)
RRF_TITLE_W  = 0.4    # weight for title-only BM25 signal in RRF

LOWERCASE        = True
REMOVE_STOPWORDS = True
STEMMING         = "porter"   # "none" | "porter" | "snowball"
MIN_TOKEN_LEN    = 2
STRIP_PUNCT      = True

# ─────────────────────────────────────────────────────────────────────────────

STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","this","that","these","those","it","its","as","if",
    "not","no","nor","so","yet","both","either","each","few","more","most",
    "other","some","such","than","then","there","when","where","which","who",
    "how","what","he","she","they","we","you","i","my","your","his","her",
    "our","their","about","up","out","into","through","during","before","after",
}

_stemmer_fn = None

def _get_stemmer():
    if STEMMING == "porter":
        from nltk.stem import PorterStemmer
        return PorterStemmer().stem
    if STEMMING == "snowball":
        from nltk.stem import SnowballStemmer
        return SnowballStemmer("english").stem
    return None

def tokenize(text: str) -> list[str]:
    global _stemmer_fn
    if _stemmer_fn is None and STEMMING != "none":
        _stemmer_fn = _get_stemmer()
    if STRIP_PUNCT:
        text = re.sub(r"[^\w\s]", " ", text)
    if LOWERCASE:
        text = text.lower()
    tokens = [t for t in text.split() if len(t) >= MIN_TOKEN_LEN]
    if REMOVE_STOPWORDS:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if _stemmer_fn:
        tokens = [_stemmer_fn(t) for t in tokens]
    return tokens


def main():
    t0 = time.time()

    corpus_file  = DATA_DIR / "corpus.json"
    queries_file = DATA_DIR / "queries.json"
    if not corpus_file.exists():
        print("[ERROR] data/ not found. Run: python prepare.py")
        raise SystemExit(1)

    print("[search] Loading data...")
    with open(corpus_file)  as f: corpus  = json.load(f)
    with open(queries_file) as f: queries = json.load(f)

    # ── Tokenize each field separately ───────────────────────────────────────
    print(f"[search] Tokenizing {len(corpus):,} documents...")
    title_tokens: dict[str, list[str]] = {}
    body_tokens:  dict[str, list[str]] = {}
    title_pos_inv: dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    body_pos_inv:  dict[str, dict[str, list]] = defaultdict(lambda: defaultdict(list))
    for did, doc in corpus.items():
        tt = tokenize(doc.get("title", ""))
        bt = tokenize(doc.get("text",  ""))
        title_tokens[did] = tt
        body_tokens[did]  = bt
        for pos, tok in enumerate(tt):
            title_pos_inv[tok][did].append(pos)
        for pos, tok in enumerate(bt):
            body_pos_inv[tok][did].append(pos)

    # ── Build per-field BM25 index ────────────────────────────────────────────
    def build_index(field_tokens):
        n      = len(field_tokens)
        dl     = {did: len(toks) for did, toks in field_tokens.items()}
        avgdl  = sum(dl.values()) / n if n else 1.0
        inv    = defaultdict(lambda: defaultdict(int))
        for did, toks in field_tokens.items():
            for tok in toks:
                inv[tok][did] += 1
        idf = {}
        for term, postings in inv.items():
            df = len(postings)
            idf[term] = math.log((n - df + 0.5) / (df + 0.5) + 1.0)
        return inv, idf, dl, avgdl

    print("[search] Building index...")
    t_inv, t_idf, t_dl, t_avgdl = build_index(title_tokens)
    b_inv, b_idf, b_dl, b_avgdl = build_index(body_tokens)

    # ── Global IDF: doc contains term if it appears in ANY field ─────────────
    n_docs = len(corpus)
    all_terms = set(t_inv.keys()) | set(b_inv.keys())
    global_idf: dict[str, float] = {}
    for term in all_terms:
        docs_with_term = set(t_inv.get(term, {}).keys()) | set(b_inv.get(term, {}).keys())
        df = len(docs_with_term)
        global_idf[term] = math.log((n_docs - df + 0.5) / (df + 0.5) + 1.0)

    # ── BM25F scoring ─────────────────────────────────────────────────────────
    def bm25f_scores(query_tokens: list[str]) -> dict[str, float]:
        candidates: set[str] = set()
        for term in query_tokens:
            candidates.update(t_inv.get(term, {}).keys())
            candidates.update(b_inv.get(term, {}).keys())

        scores = {}
        for did in candidates:
            score = 0.0
            for term in query_tokens:
                idf = global_idf.get(term, 0.0)

                tf_t   = t_inv.get(term, {}).get(did, 0)
                norm_t = 1 - B_TITLE + B_TITLE * t_dl.get(did, t_avgdl) / t_avgdl
                wtf_t  = TITLE_WEIGHT * tf_t / norm_t if norm_t > 0 else 0

                tf_b   = b_inv.get(term, {}).get(did, 0)
                norm_b = 1 - B_BODY + B_BODY * b_dl.get(did, b_avgdl) / b_avgdl
                wtf_b  = tf_b / norm_b if norm_b > 0 else 0

                wtf  = wtf_t + wtf_b
                tf_n = (wtf * (K1 + 1)) / (wtf + K1) + DELTA
                score += idf * tf_n

            scores[did] = score
        return scores

    # ── RM3-style pseudo-relevance feedback ──────────────────────────────────
    def expand_query(q_toks: list[str], initial_scores: dict[str, float]) -> dict[str, float]:
        """Mine top-k docs to find expansion terms; return term weights."""
        # Sort and take top FB_DOCS docs
        top_docs = sorted(initial_scores.items(), key=lambda x: x[1], reverse=True)[:FB_DOCS]
        if not top_docs:
            return {}

        # Normalize scores to sum to 1 (doc weights)
        score_sum = sum(s for _, s in top_docs)
        if score_sum == 0:
            return {}

        q_set = set(q_toks)
        term_weights: dict[str, float] = defaultdict(float)

        for did, doc_score in top_docs:
            doc_weight = doc_score / score_sum
            # Combine title + body tokens for term extraction
            # Weight title tokens to match BM25F title field importance
            all_toks = title_tokens[did] * int(TITLE_WEIGHT) + body_tokens[did]
            dl = len(all_toks)
            if dl == 0:
                continue
            # Count term frequencies in this doc
            tf_map: dict[str, int] = defaultdict(int)
            for tok in all_toks:
                tf_map[tok] += 1
            # P(w|d) = tf / dl, weighted by doc relevance weight
            # Include original query terms so confirmed terms get amplified (full RM3)
            for tok, tf in tf_map.items():
                idf = global_idf.get(tok, 0.0)
                term_weights[tok] += doc_weight * (tf / dl) * idf

        # Pick top FB_TERMS (may include original query terms if they rank highly)
        top_terms = sorted(term_weights.items(), key=lambda x: x[1], reverse=True)[:FB_TERMS]
        return dict(top_terms)

    def bm25f_scores_weighted(query_weights: dict[str, float]) -> dict[str, float]:
        """Like bm25f_scores but each query term has a weight."""
        candidates: set[str] = set()
        for term in query_weights:
            candidates.update(t_inv.get(term, {}).keys())
            candidates.update(b_inv.get(term, {}).keys())

        scores = {}
        for did in candidates:
            score = 0.0
            for term, q_weight in query_weights.items():
                idf = global_idf.get(term, 0.0)

                tf_t   = t_inv.get(term, {}).get(did, 0)
                norm_t = 1 - B_TITLE + B_TITLE * t_dl.get(did, t_avgdl) / t_avgdl
                wtf_t  = TITLE_WEIGHT * tf_t / norm_t if norm_t > 0 else 0

                tf_b   = b_inv.get(term, {}).get(did, 0)
                norm_b = 1 - B_BODY + B_BODY * b_dl.get(did, b_avgdl) / b_avgdl
                wtf_b  = tf_b / norm_b if norm_b > 0 else 0

                wtf  = wtf_t + wtf_b
                tf_n = (wtf * (K1 + 1)) / (wtf + K1) + DELTA
                score += q_weight * idf * tf_n

            scores[did] = score
        return scores

    # ── Proximity scoring helpers ─────────────────────────────────────────────
    def _min_sorted_dist(a: list, b: list) -> int:
        """Minimum distance between elements of two position lists (both sorted)."""
        i = j = 0
        best = 10**9
        while i < len(a) and j < len(b):
            d = abs(a[i] - b[j])
            if d < best:
                best = d
            if a[i] <= b[j]:
                i += 1
            else:
                j += 1
        return best

    def proximity_bonus(q_toks: list[str], candidate_docs: set[str]) -> dict[str, float]:
        """For each candidate doc, score how closely query terms co-occur."""
        unique_q = list(set(q_toks))
        if len(unique_q) < 2:
            return {}
        bonus: dict[str, float] = {}
        for did in candidate_docs:
            score = 0.0
            for i in range(len(unique_q)):
                t1 = unique_q[i]
                for j in range(i + 1, len(unique_q)):
                    t2 = unique_q[j]
                    bp1 = body_pos_inv.get(t1, {}).get(did)
                    bp2 = body_pos_inv.get(t2, {}).get(did)
                    if bp1 and bp2:
                        score += 1.0 / (1 + _min_sorted_dist(bp1, bp2))
                    tp1 = title_pos_inv.get(t1, {}).get(did)
                    tp2 = title_pos_inv.get(t2, {}).get(did)
                    if tp1 and tp2:
                        score += TITLE_WEIGHT / (1 + _min_sorted_dist(tp1, tp2))
            if score > 0.0:
                bonus[did] = score
        return bonus

    # ── Run all queries with PRF ──────────────────────────────────────────────
    print(f"[search] Running {len(queries):,} queries with PRF "
          f"(FB_DOCS={FB_DOCS}, FB_TERMS={FB_TERMS}, alpha={FB_ALPHA})...")
    rankings: dict[str, list[str]] = {}
    for qid, qtext in queries.items():
        q_toks = tokenize(qtext)
        if not q_toks:
            rankings[qid] = []
            continue

        # Pass 1: initial BM25F scoring
        initial_scores = bm25f_scores(q_toks)

        # Pass 2: expand query with feedback terms
        expansion_terms = expand_query(q_toks, initial_scores)

        if expansion_terms:
            # Build combined query weights
            # Original terms: weight FB_ALPHA, scaled by IDF (rarer terms get more weight)
            orig_idf = {t: global_idf.get(t, 0.0) for t in q_toks}
            orig_idf_total = sum(orig_idf.values())
            exp_total = sum(expansion_terms.values())
            exp_scale = (1.0 - FB_ALPHA) / exp_total if exp_total > 0 else 0

            query_weights: dict[str, float] = {}
            if orig_idf_total > 0:
                orig_scale = FB_ALPHA / orig_idf_total
                for t in q_toks:
                    query_weights[t] = query_weights.get(t, 0) + orig_idf[t] * orig_scale
            else:
                orig_weight = FB_ALPHA / len(q_toks)
                for t in q_toks:
                    query_weights[t] = query_weights.get(t, 0) + orig_weight
            for t, w in expansion_terms.items():
                query_weights[t] = query_weights.get(t, 0) + w * exp_scale

            final_scores = bm25f_scores_weighted(query_weights)
        else:
            final_scores = initial_scores

        # Pass 3: proximity bonus
        prox = proximity_bonus(q_toks, set(final_scores.keys()))
        if prox:
            top_bm25 = max(final_scores.values())
            top_prox = max(prox.values())
            if top_prox > 0:
                scale = PROX_WEIGHT * top_bm25 / top_prox
                for did, pb in prox.items():
                    if did in final_scores:
                        final_scores[did] += scale * pb

        rankings[qid] = [did for did, _ in
                         sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:1000]]

    with open(OUT_FILE, "w") as f:
        json.dump(rankings, f)

    print(f"[search] Done in {time.time()-t0:.2f}s — {OUT_FILE.name} written.")


if __name__ == "__main__":
    main()
