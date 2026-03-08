from pathlib import Path
import pickle

import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sentence_transformers import SentenceTransformer


def load_and_embed():
    data_dir = Path("data")
    embeddings_path = data_dir / "embeddings.npy"
    documents_path = data_dir / "documents.pkl"

    if embeddings_path.exists() and documents_path.exists():
        print("Loading cached documents and embeddings...")
        embeddings = np.load(embeddings_path)
        with documents_path.open("rb") as f:
            docs = pickle.load(f)
        print(f"Loaded {len(docs)} documents from cache")
        return docs, embeddings

    print("Loading dataset...")

    data = fetch_20newsgroups(
        subset="all",
        remove=("headers", "footers", "quotes")
    )

    docs = data.data

    print(f"Loaded {len(docs)} documents")

    print("Loading embedding model...")
    model = SentenceTransformer("all-MiniLM-L6-v2")

    print("Generating embeddings (this may take a few minutes)...")

    embeddings = model.encode(
        docs,
        show_progress_bar=True
    )

    print("Embeddings created!")

    data_dir.mkdir(parents=True, exist_ok=True)
    np.save(embeddings_path, embeddings)
    with documents_path.open("wb") as f:
        pickle.dump(docs, f)
    print("Saved documents and embeddings to data/")

    return docs, embeddings

if __name__ == "__main__":
    docs, embeddings = load_and_embed()
    print(len(docs))
    print(embeddings.shape)