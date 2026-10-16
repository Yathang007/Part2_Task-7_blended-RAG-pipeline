import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer
from src.fusion import reciprocal_rank_fusion

class HybridRetriever:
    """BM25 + dense FAISS retrieval combined with Reciprocal Rank Fusion."""
    def __init__(self, chunks, embedding_model="sentence-transformers/all-MiniLM-L6-v2"):
        if not chunks:
            raise ValueError("At least one text chunk is required")
        self.chunks = [dict(chunk, id=str(chunk.get("chunk_id", i))) for i, chunk in enumerate(chunks)]
        self.bm25 = BM25Okapi([self._tokenize(chunk["text"]) for chunk in self.chunks])
        self.encoder = SentenceTransformer(embedding_model)
        vectors = self.encoder.encode(
            [chunk["text"] for chunk in self.chunks],
            convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.count = len(vectors)

    @staticmethod
    def _tokenize(text):
        return str(text).lower().split()

    def _result(self, index, score, method):
        result = dict(self.chunks[index])
        result["id"] = str(result.get("id", index))
        result["retrieval_score"] = float(score)
        result["retrieval_method"] = method
        return result

    def keyword_search(self, query, top_k=8):
        scores = self.bm25.get_scores(self._tokenize(query))
        indices = np.argsort(scores)[::-1][:min(top_k, self.count)]
        return [self._result(int(i), scores[i], "bm25") for i in indices if scores[i] > 0]

    def semantic_search(self, query, top_k=8):
        vector = self.encoder.encode(
            [query], convert_to_numpy=True, normalize_embeddings=True, show_progress_bar=False
        ).astype("float32")
        scores, indices = self.index.search(vector, min(top_k, self.count))
        return [self._result(int(i), score, "dense")
                for score, i in zip(scores[0], indices[0]) if i >= 0]

    def blended_search(self, query, top_k=5, candidate_k=12, rrf_k=60):
        keyword = self.keyword_search(query, candidate_k)
        semantic = self.semantic_search(query, candidate_k)
        results = reciprocal_rank_fusion([keyword, semantic], k=rrf_k, top_n=top_k)
        for result in results:
            result["retrieval_method"] = "hybrid_rrf"
        return results
