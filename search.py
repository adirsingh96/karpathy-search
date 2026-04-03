"""
search.py — Search algorithm for btter_bm25.

THE FILE THE AGENT REWRITES.

Hard interface contract (do not break these):
  INPUT  : data/corpus.json   → {doc_id: {"title": str, "text": str}}
  INPUT  : data/queries.json  → {query_id: query_text}
  OUTPUT : rankings.json      → {query_id: [doc_id, ...]}   (ranked best-first, ≤1000 per query)

Everything else is fair game — rewrite the algorithm however you like.
Current approach: BM25 with separate title / body field weighting (BM25F-style).
"""

import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

ROOT     = Path(__file__).parent
DATA_DIR = ROOT / "data"
OUT_FILE = ROOT / "rankings.json"

# ── Parameters (agent: change these OR rewrite the whole algorithm) ───────────

K1           = 1.5    # TF saturation
B_TITLE      = 0.4    # length norm for title field  (shorter → lower B is better)
B_BODY       = 0.75   # length norm for body field
TITLE_WEIGHT = 3.0    # how much more a title match counts vs. a body match
DELTA        = 0.0    # BM25+ floor (0 = standard BM25)

LOWERCASE        = True
REMOVE_STOPWORDS = True
STEMMING         = "none"   # "none" | "porter" | "snowball"
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
    for did, doc in corpus.items():
        title_tokens[did] = tokenize(doc.get("title", ""))
        body_tokens[did]  = tokenize(doc.get("text",  ""))

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

    # ── BM25F scoring ─────────────────────────────────────────────────────────
    def bm25f_scores(query_tokens: list[str]) -> dict[str, float]:
        # Collect candidate docs from both fields
        candidates: set[str] = set()
        for term in query_tokens:
            candidates.update(t_inv.get(term, {}).keys())
            candidates.update(b_inv.get(term, {}).keys())

        scores = {}
        for did in candidates:
            score = 0.0
            for term in query_tokens:
                idf = b_idf.get(term, t_idf.get(term, 0.0))  # use body IDF (larger corpus)

                # Title contribution
                tf_t   = t_inv.get(term, {}).get(did, 0)
                norm_t = 1 - B_TITLE + B_TITLE * t_dl.get(did, t_avgdl) / t_avgdl
                wtf_t  = TITLE_WEIGHT * tf_t / norm_t if norm_t > 0 else 0

                # Body contribution
                tf_b   = b_inv.get(term, {}).get(did, 0)
                norm_b = 1 - B_BODY + B_BODY * b_dl.get(did, b_avgdl) / b_avgdl
                wtf_b  = tf_b / norm_b if norm_b > 0 else 0

                # Combined pseudo-TF
                wtf  = wtf_t + wtf_b
                tf_n = (wtf * (K1 + 1)) / (wtf + K1) + DELTA
                score += idf * tf_n

            scores[did] = score
        return scores

    # ── Run all queries ───────────────────────────────────────────────────────
    print(f"[search] Running {len(queries):,} queries "
          f"(K1={K1}, B_title={B_TITLE}, B_body={B_BODY}, title_weight={TITLE_WEIGHT})...")
    rankings: dict[str, list[str]] = {}
    for qid, qtext in queries.items():
        q_toks = tokenize(qtext)
        if not q_toks:
            rankings[qid] = []
            continue
        scored = bm25f_scores(q_toks)
        rankings[qid] = [did for did, _ in
                         sorted(scored.items(), key=lambda x: x[1], reverse=True)[:1000]]

    with open(OUT_FILE, "w") as f:
        json.dump(rankings, f)

    print(f"[search] Done in {time.time()-t0:.2f}s — {OUT_FILE.name} written.")


if __name__ == "__main__":
    main()
