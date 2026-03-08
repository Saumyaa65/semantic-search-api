from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np


class SemanticCache:
	"""In-memory semantic cache using cosine similarity on embeddings."""

	def __init__(self, similarity_threshold: float = 0.85) -> None:
		if not 0.0 <= similarity_threshold <= 1.0:
			raise ValueError("similarity_threshold must be between 0 and 1")

		self.similarity_threshold = similarity_threshold
		self._entries: List[Dict[str, Any]] = []
		self.hit_count = 0
		self.miss_count = 0

	def add(self, query_embedding: np.ndarray, result: Any) -> None:
		"""Store a query embedding and its corresponding result."""
		embedding = self._to_1d_float32(query_embedding)
		self._entries.append({"embedding": embedding, "result": result})

	def search(self, query_embedding: np.ndarray) -> Optional[Any]:
		"""Return cached result when best cosine similarity is above threshold."""
		if not self._entries:
			self.miss_count += 1
			return None

		query = self._to_1d_float32(query_embedding)
		query_norm = np.linalg.norm(query)

		if query_norm == 0.0:
			self.miss_count += 1
			return None

		best_similarity = -1.0
		best_result: Optional[Any] = None

		for entry in self._entries:
			candidate = entry["embedding"]
			candidate_norm = np.linalg.norm(candidate)

			if candidate_norm == 0.0:
				continue

			similarity = float(np.dot(query, candidate) / (query_norm * candidate_norm))

			if similarity > best_similarity:
				best_similarity = similarity
				best_result = entry["result"]

		if best_similarity > self.similarity_threshold:
			self.hit_count += 1
			return best_result

		self.miss_count += 1
		return None

	def stats(self) -> Dict[str, int]:
		"""Return cache statistics."""
		return {
			"hit_count": self.hit_count,
			"miss_count": self.miss_count,
			"total_entries": len(self._entries),
		}

	def clear(self) -> None:
		"""Clear cached entries and reset counters."""
		self._entries.clear()
		self.hit_count = 0
		self.miss_count = 0

	@staticmethod
	def _to_1d_float32(embedding: np.ndarray) -> np.ndarray:
		"""Normalize input to a 1D float32 numpy array."""
		arr = np.asarray(embedding, dtype=np.float32)

		if arr.ndim == 2 and arr.shape[0] == 1:
			arr = arr.reshape(-1)
		elif arr.ndim != 1:
			raise ValueError("embedding must have shape (dim,) or (1, dim)")

		return arr

if __name__ == "__main__":
    import numpy as np

    cache = SemanticCache()

    q1 = np.random.rand(384)
    result = "example result"

    cache.add(q1, result)

    found = cache.search(q1)

    print(found)
    print(cache.stats())