from __future__ import annotations

from typing import Optional

import faiss
import numpy as np


class FaissSearcher:
	"""Build and query a FAISS IndexFlatL2 over document embeddings."""

	def __init__(self, embeddings: np.ndarray) -> None:
		if embeddings.ndim != 2:
			raise ValueError("embeddings must be a 2D numpy array of shape (n_docs, dim)")
		if embeddings.size == 0:
			raise ValueError("embeddings must not be empty")

		self.embeddings = np.asarray(embeddings, dtype=np.float32)
		self.dim = self.embeddings.shape[1]
		self.index: Optional[faiss.IndexFlatL2] = None

		self._build_index()

	def _build_index(self) -> None:
		"""Create an IndexFlatL2 and add all document embeddings."""
		self.index = faiss.IndexFlatL2(self.dim)
		self.index.add(self.embeddings)

	def search(self, query_embedding: np.ndarray, k: int = 3) -> np.ndarray:
		"""Return indices of the k nearest documents for a query embedding."""
		if self.index is None:
			raise RuntimeError("FAISS index is not initialized")
		if k <= 0:
			raise ValueError("k must be a positive integer")

		query = np.asarray(query_embedding, dtype=np.float32)

		if query.ndim == 1:
			query = query.reshape(1, -1)
		elif query.ndim != 2 or query.shape[0] != 1:
			raise ValueError("query_embedding must have shape (dim,) or (1, dim)")

		if query.shape[1] != self.dim:
			raise ValueError(f"query dimension {query.shape[1]} does not match index dimension {self.dim}")

		_, indices = self.index.search(query, k)
		return indices[0]


def build_index(embeddings: np.ndarray) -> FaissSearcher:
	"""Build and return a FAISS IndexFlatL2 searcher for document embeddings."""
	return FaissSearcher(embeddings)

if __name__ == "__main__":
    from embeddings import load_and_embed
    docs, embeddings = load_and_embed()

    index = FaissSearcher(embeddings)

    results = index.search(embeddings[0], k=3)
    print(results)