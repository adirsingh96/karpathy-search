"""
search.py — BM25 search engine for btter_bm25.

THIS IS THE FILE YOU (THE AGENT) MODIFY.
Only edit parameters inside the TUNABLE PARAMETERS section.
Do not change the BM25 implementation or data loading logic.
"""

import json
import math
import re
import time
from collections import defaultdict
from pathlib import Path

# ===== BEGIN TUNABLE PARAMETERS =====

# -- BM25 core --
K1    = 1.5     # term frequency saturation (0.5 – 3.0)
B     = 0.75    # length normalization      (0.0 – 1.0); 0 = off, 1 = full
DELTA = 0.0     # BM25+ floor term          (0.0 – 1.0); 0 = standard BM25

# -- Preprocessing --
LOWERCASE         = True    # lowercase all tokens
REMOVE_STOPWORDS  = True    # remove common English stopwords
STEMMING          = "none"  # "none" | "porter" | "snowball"
MIN_TOKEN_LEN     = 2       # drop tokens shorter than this
STRIP_PUNCTUATION = True    # remove punctuation before tokenizing

# -- Indexing --
USE_BIGRAMS    = False  # also index adjacent word pairs (e.g. "information retrieval")
BIGRAM_WEIGHT  = 0.3    # relative weight of bigram scores vs. unigram scores

# -- Query expansion (pseudo-relevance feedback) --
QUERY_EXPANSION = False   # re-rank using terms from top retrieved docs
EXPANSION_DOCS  = 3       # number of top docs to mine for expansion terms
EXPANSION_TERMS = 10      # number of new terms to add to query

# ===== END TUNABLE PARAMETERS =====


# ── Paths ─────────────────────────────────────────────────────────────────────
ROOT      = Path(__file__).parent
DATA_DIR  = ROOT / "data"
OUT_FILE  = ROOT / "rankings.json"   # written by this script, read by eval.py

# ── Stopwords ──────────────────────────────────────────────────────────────────
STOPWORDS = {
    "a","an","the","and","or","but","in","on","at","to","for","of","with",
    "by","from","is","are","was","were","be","been","being","have","has",
    "had","do","does","did","will","would","could","should","may","might",
    "shall","can","need","this","that","these","those","it","its","as","if",
    "not","no","nor","so","yet","both","either","each","few","more","most",
    "other","some","such","than","then","there","when","where","which","who",
    "how","what","he","she","they","we","you","i","my","your","his","her",
    "our","their","about","up","out","into","through","during","before","after",
}


# ── Tokenizer ──────────────────────────────────────────────────────────────────

def _get_stemmer():
    if STEMMING == "porter":
        from nltk.stem import PorterStemmer
        return PorterStemmer().stem
    if STEMMING == "snowball":
        from nltk.stem import SnowballStemmer
        return SnowballStemmer("english").stem
    return None

_stemmer = None

def tokenize(text: str) -> list[str]:
    global _stemmer
    if _stemmer is None and STEMMING != "none":
        _stemmer = _get_stemmer()

    if STRIP_PUNCTUATION:
        text = re.sub(r"[^\w\s]", " ", text)

    if LOWERCASE:
        text = text.lower()

    tokens = text.split()

    if MIN_TOKEN_LEN > 1:
        tokens = [t for t in tokens if len(t) >= MIN_TOKEN_LEN]

    if REMOVE_STOPWORDS:
        tokens = [t for t in tokens if t not in STOPWORDS]

    if _stemmer:
        tokens = [_stemmer(t) for t in tokens]

    return tokens


def make_tokens(text: str) -> list[str]:
    """Return unigrams (and optionally bigrams) for a piece of text."""
    uni = tokenize(text)
    if not USE_BIGRAMS:
        return uni
    bi = [f"{uni[i]}_{uni[i+1]}" for i in range(len(uni) - 1)]
    return uni + bi


# ── BM25 index ────────────────────────────────────────────────────────────────

class BM25Index:
    def __init__(self, docs: dict[str, list[str]]):
        """
        docs: {doc_id: token_list}
        """
        self.k1 = K1
        self.b  = B
        self.delta = DELTA
        self.n  = len(docs)

        # document lengths
        self.dl: dict[str, int] = {did: len(toks) for did, toks in docs.items()}
        self.avgdl = sum(self.dl.values()) / self.n if self.n else 1.0

        # inverted index: term → {doc_id: tf}
        self.inv: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
        for did, toks in docs.items():
            for tok in toks:
                self.inv[tok][did] += 1

        # IDF for each term
        self.idf: dict[str, float] = {}
        for term, postings in self.inv.items():
            df = len(postings)
            self.idf[term] = math.log((self.n - df + 0.5) / (df + 0.5) + 1.0)

    def score(self, query_tokens: list[str], doc_id: str) -> float:
        dl   = self.dl.get(doc_id, self.avgdl)
        norm = 1 - self.b + self.b * dl / self.avgdl
        s    = 0.0
        for term in query_tokens:
            if term not in self.inv or doc_id not in self.inv[term]:
                continue
            tf  = self.inv[term][doc_id]
            idf = self.idf.get(term, 0.0)
            # BM25+ when DELTA > 0
            tf_norm = (tf * (self.k1 + 1)) / (tf + self.k1 * norm) + self.delta
            s += idf * tf_norm
        return s

    def search(self, query_tokens: list[str], top_k: int = 1000) -> list[tuple[str, float]]:
        """Return top_k (doc_id, score) pairs sorted by score descending."""
        # Only score docs that contain at least one query term
        candidate_docs: set[str] = set()
        for term in query_tokens:
            if term in self.inv:
                candidate_docs.update(self.inv[term].keys())

        scores = [(did, self.score(query_tokens, did)) for did in candidate_docs]
        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]


# ── Pseudo-relevance feedback ──────────────────────────────────────────────────

def expand_query(
    original_tokens: list[str],
    top_docs: list[tuple[str, float]],
    all_doc_tokens: dict[str, list[str]],
    index: BM25Index,
) -> list[str]:
    """
    Add EXPANSION_TERMS high-IDF terms from EXPANSION_DOCS top-ranked docs.
    """
    term_freq: dict[str, int] = defaultdict(int)
    for did, _ in top_docs[:EXPANSION_DOCS]:
        for tok in all_doc_tokens.get(did, []):
            term_freq[tok] += 1

    # Score expansion candidates by tf × idf, exclude original query terms
    original_set = set(original_tokens)
    candidates = [
        (tok, freq * index.idf.get(tok, 0.0))
        for tok, freq in term_freq.items()
        if tok not in original_set
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    new_terms = [tok for tok, _ in candidates[:EXPANSION_TERMS]]
    return original_tokens + new_terms


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    t0 = time.time()

    # Load corpus
    corpus_file = DATA_DIR / "corpus.json"
    queries_file = DATA_DIR / "queries.json"
    if not corpus_file.exists() or not queries_file.exists():
        print("[ERROR] data/ not found. Run:  python prepare.py")
        raise SystemExit(1)

    print("[search] Loading corpus...")
    with open(corpus_file) as f:
        corpus_raw = json.load(f)   # {doc_id: {"title": ..., "text": ...}}

    with open(queries_file) as f:
        queries_raw = json.load(f)  # {query_id: query_text}

    # Tokenize corpus
    print(f"[search] Tokenizing {len(corpus_raw):,} documents "
          f"(stemming={STEMMING!r}, stopwords={REMOVE_STOPWORDS}, bigrams={USE_BIGRAMS})...")
    doc_tokens: dict[str, list[str]] = {}
    for did, doc in corpus_raw.items():
        text = (doc.get("title", "") + " " + doc.get("text", "")).strip()
        doc_tokens[did] = make_tokens(text)

    # Build index
    print("[search] Building BM25 index...")
    index = BM25Index(doc_tokens)
    print(f"[search] Index: {len(index.inv):,} unique terms, avgdl={index.avgdl:.1f}")

    # Run queries
    print(f"[search] Running {len(queries_raw):,} queries (K1={K1}, B={B}, DELTA={DELTA})...")
    rankings: dict[str, list[str]] = {}

    for qid, qtext in queries_raw.items():
        q_tokens = make_tokens(qtext)
        if not q_tokens:
            rankings[qid] = []
            continue

        results = index.search(q_tokens, top_k=1000)

        if QUERY_EXPANSION and results:
            q_tokens_expanded = expand_query(q_tokens, results, doc_tokens, index)
            results = index.search(q_tokens_expanded, top_k=1000)

        rankings[qid] = [did for did, _ in results]

    # Save rankings
    with open(OUT_FILE, "w") as f:
        json.dump(rankings, f)

    elapsed = time.time() - t0
    print(f"[search] Done in {elapsed:.2f}s — rankings saved to {OUT_FILE.name}")


if __name__ == "__main__":
    main()
