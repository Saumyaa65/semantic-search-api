from __future__ import annotations

from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException
from openai import embeddings
from pydantic import BaseModel, Field
from sentence_transformers import SentenceTransformer

from cache import SemanticCache
import embeddings
from embeddings import load_and_embed
from search import FaissSearcher, build_index
from clustering import cluster_embeddings

class QueryRequest(BaseModel):
	query: str = Field(..., min_length=1)


class QueryResponse(BaseModel):
	query: str
	cache_hit: bool
	results: List[str]
	clusters: List[int]


app = FastAPI(title="Semantic Search API")

documents: List[str] = []
searcher: Optional[FaissSearcher] = None
query_model: Optional[SentenceTransformer] = None
semantic_cache = SemanticCache()
cluster_probs = None

@app.get("/")
def home():
    return {"message": "Semantic Search API running. Visit /docs for API UI"}

@app.on_event("startup")
def startup_event() -> None:
    global documents, searcher, query_model, cluster_probs

    documents, embeddings = load_and_embed()
    embeddings = np.asarray(embeddings, dtype=np.float32)

    cluster_probs, gmm_model = cluster_embeddings(embeddings)

    searcher = build_index(embeddings)

    query_model = SentenceTransformer("all-MiniLM-L6-v2")


@app.post("/query", response_model=QueryResponse)
def query_documents(payload: QueryRequest) -> QueryResponse:
	if searcher is None or query_model is None or cluster_probs is None:
		raise HTTPException(status_code=503, detail="Service is still initializing")

	query_embedding = query_model.encode(payload.query, convert_to_numpy=True)

	cached_result = semantic_cache.search(query_embedding)
	if cached_result is not None:
		results, clusters = cached_result
		return QueryResponse(query=payload.query, cache_hit=True, results=results, clusters=clusters)

	top_indices = searcher.search(query_embedding, k=3)
	clusters = [int(np.argmax(cluster_probs[i])) for i in top_indices]
	results = [documents[i] for i in top_indices]
	semantic_cache.add(query_embedding, (results, clusters))
	return QueryResponse(query=payload.query, cache_hit=False, results=results, clusters=clusters)


@app.get("/cache/stats")
def get_cache_stats() -> dict:
	return semantic_cache.stats()


@app.delete("/cache")
def clear_cache() -> dict:
	semantic_cache.clear()
	return {"message": "Cache cleared"}
