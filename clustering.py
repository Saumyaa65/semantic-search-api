from __future__ import annotations

from typing import Tuple

import numpy as np
from sklearn.mixture import GaussianMixture


def cluster_embeddings(
	embeddings: np.ndarray,
	n_clusters: int = 10,
) -> Tuple[np.ndarray, GaussianMixture]:
	
	X = np.asarray(embeddings, dtype=np.float32)

	if X.ndim != 2:
		raise ValueError("embeddings must be a 2D numpy array")
	if X.shape[0] == 0:
		raise ValueError("embeddings must not be empty")
	if n_clusters <= 0:
		raise ValueError("n_clusters must be a positive integer")

	model = GaussianMixture(n_components=n_clusters, random_state=42)
	model.fit(X)
	cluster_probabilities = model.predict_proba(X)

	return cluster_probabilities, model

if __name__ == "__main__":

    from embeddings import load_and_embed

    docs, embeddings = load_and_embed()

    probs, model = cluster_embeddings(embeddings)

    print("Shape:", probs.shape)
    print("Example:", probs[0])