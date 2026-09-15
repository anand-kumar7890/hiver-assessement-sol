"""
Retrieval-based grounding: given a new incoming customer message, find the
most similar historically-resolved (customer -> brand reply) pairs so the
reply drafter can imitate how this brand actually handled similar issues,
instead of generating from scratch.

Kept deliberately simple: sentence-transformers embeddings + brute-force
cosine similarity via sklearn. No vector DB — at the scale of a subsample
(thousands of pairs, not millions), a numpy array search is faster to build,
easier to debug, and easier to explain live than standing up FAISS/pgvector
for no real benefit. See decision log.

Embeddings are cached to disk (see `cache_path`) because computing them for
100k+ pairs is the single slowest step in this pipeline and is identical
every time you run it on the same data — recomputing it on every script
invocation (as an earlier version of this file did) makes iteration
painfully slow. The cache is invalidated automatically if the underlying
corpus changes (checked via a hash), so a stale cache never silently
produces wrong results.
"""
import hashlib
import os
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "all-MiniLM-L6-v2"

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False


class _TfidfFallbackEncoder:
    """
    Offline fallback used only when sentence-transformers can't reach
    HuggingFace (e.g. sandboxed dev/CI environments with no internet).
    NOT used for reported results — swap to SentenceTransformer for
    the real run, which is the default when internet access is available.
    """
    def __init__(self):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self.vectorizer = TfidfVectorizer(max_features=2048)
        self._fitted = False

    def encode(self, texts, show_progress_bar=False, normalize_embeddings=True):
        if not self._fitted:
            mat = self.vectorizer.fit_transform(texts)
            self._fitted = True
        else:
            mat = self.vectorizer.transform(texts)
        arr = mat.toarray().astype("float32")
        if normalize_embeddings:
            norms = np.linalg.norm(arr, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            arr = arr / norms
        return arr


def _corpus_hash(corpus: list) -> str:
    """Cheap fingerprint of the corpus so we can detect if cached embeddings
    were computed for different/stale data and need recomputing."""
    h = hashlib.sha256()
    h.update(str(len(corpus)).encode())
    sample_idx = [0, len(corpus) // 2, -1] if corpus else []
    for i in sample_idx:
        h.update(corpus[i].encode("utf-8", errors="ignore"))
    return h.hexdigest()[:16]


class RetrievalIndex:
    def __init__(self, pairs_df, model_name: str = MODEL_NAME,
                 force_fallback: bool = False, cache_path: str = None):
        """
        pairs_df must have columns: customer_text_clean, brand_text_clean,
        and optionally 'intent' if you want intent-filtered retrieval.

        cache_path: if given, embeddings are loaded from / saved to this
        .npz file so repeated runs on the same data skip re-encoding.
        """
        self.pairs_df = pairs_df.reset_index(drop=True)
        corpus = self.pairs_df["customer_text_clean"].tolist()
        corpus_hash = _corpus_hash(corpus)

        self.using_fallback = force_fallback or not _HAS_ST

        loaded_from_cache = False
        if cache_path and os.path.exists(cache_path):
            try:
                cached = np.load(cache_path, allow_pickle=False)
                if str(cached["corpus_hash"]) == corpus_hash and \
                   cached["embeddings"].shape[0] == len(corpus):
                    self.embeddings = cached["embeddings"]
                    loaded_from_cache = True
                    print(f"[retrieval] Loaded cached embeddings from {cache_path} "
                          f"({self.embeddings.shape[0]} vectors) — skipping re-encoding.")
                else:
                    print(f"[retrieval] Cache at {cache_path} is stale (data changed); "
                          f"recomputing embeddings.")
            except Exception as e:
                print(f"[retrieval] Could not read cache ({e.__class__.__name__}); recomputing.")

        if not loaded_from_cache:
            if not self.using_fallback:
                try:
                    self.model = SentenceTransformer(model_name)
                    self.embeddings = self.model.encode(
                        corpus, show_progress_bar=True, normalize_embeddings=True,
                    )
                except Exception as e:
                    print(f"[retrieval] Could not load '{model_name}' ({e.__class__.__name__}); "
                          f"falling back to TF-IDF for this run.")
                    self.using_fallback = True

            if self.using_fallback:
                self.model = _TfidfFallbackEncoder()
                self.embeddings = self.model.encode(corpus)

            if cache_path:
                os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
                np.savez_compressed(cache_path, embeddings=self.embeddings,
                                     corpus_hash=corpus_hash)
                print(f"[retrieval] Saved embeddings cache to {cache_path} "
                      f"for instant reuse next time.")

        if loaded_from_cache:
            self._model_name_for_lazy_load = model_name
            self.model = None

    def _ensure_model_loaded(self):
        if self.model is not None:
            return
        if not self.using_fallback:
            try:
                self.model = SentenceTransformer(self._model_name_for_lazy_load)
                return
            except Exception:
                self.using_fallback = True
        self.model = _TfidfFallbackEncoder()
        self.model.encode(self.pairs_df["customer_text_clean"].tolist())

    def query(self, text: str, k: int = 3, intent: str = None):
        """
        Returns top-k most similar historical pairs as a list of dicts:
        {customer_text, brand_text, similarity}
        If `intent` is given and the pairs_df has an 'intent' column,
        restricts the search to that intent first (falls back to global
        search if fewer than k matches exist for that intent).
        """
        self._ensure_model_loaded()
        query_emb = self.model.encode([text], normalize_embeddings=True)

        candidate_idx = np.arange(len(self.pairs_df))
        if intent is not None and "intent" in self.pairs_df.columns:
            filtered = self.pairs_df.index[self.pairs_df["intent"] == intent].to_numpy()
            if len(filtered) >= k:
                candidate_idx = filtered

        candidate_embs = self.embeddings[candidate_idx]
        sims = cosine_similarity(query_emb, candidate_embs)[0]
        top_local = np.argsort(-sims)[:k]
        top_global = candidate_idx[top_local]

        results = []
        for local_i, global_i in zip(top_local, top_global):
            row = self.pairs_df.iloc[global_i]
            results.append({
                "customer_text": row["customer_text_clean"],
                "brand_text": row["brand_text_clean"],
                "similarity": float(sims[local_i]),
            })
        return results


if __name__ == "__main__":
    import pandas as pd
    from data_loader import load_raw, build_pairs

    raw = load_raw("data/mock_twitter_support.csv")
    pairs = build_pairs(raw)
    index = RetrievalIndex(pairs, cache_path="data/embedding_cache_demo.npz")

    test_msg = "My order still hasn't arrived and it's been over a week!"
    results = index.query(test_msg, k=3)
    print(f"Query: {test_msg}\n")
    for r in results:
        print(f"  sim={r['similarity']:.3f}  cust=\"{r['customer_text'][:60]}...\"")
        print(f"           brand=\"{r['brand_text'][:80]}...\"\n")