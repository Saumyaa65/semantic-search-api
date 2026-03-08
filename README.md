# Semantic Search API with Fuzzy Clustering and Semantic Cache

## Overview

This project implements a semantic search system using document embeddings, vector search, fuzzy clustering, and a semantic caching layer. The system exposes a FastAPI interface that allows users to query documents using natural language.

Key features include:

* SentenceTransformer embeddings
* FAISS vector similarity search
* Fuzzy document clustering using Gaussian Mixture Models
* Semantic query caching using cosine similarity
* FastAPI service for querying documents

---

## System Design Overview

The system is designed as a semantic search pipeline that retrieves relevant documents while minimizing repeated computation through a semantic caching layer.

The processing pipeline consists of the following stages:

1. **Document Embedding**

   * Each document from the dataset is converted into a dense vector representation using a transformer-based embedding model.

2. **Fuzzy Clustering**

   * Documents are grouped using a Gaussian Mixture Model (GMM), allowing each document to belong to multiple clusters with probability distributions.

3. **Vector Search**

   * All document embeddings are indexed using FAISS to enable fast similarity search.

4. **Semantic Cache**

   * Previous query embeddings and results are stored. If a new query is semantically similar to a cached query, results are returned directly without recomputation.

5. **API Layer**

   * A FastAPI service exposes endpoints for querying the system and managing cache operations.


## Features

### Semantic Search

User queries are embedded using the same transformer model as the dataset and matched against a FAISS vector index to retrieve the most relevant documents.

### Fuzzy Clustering

Documents are clustered using a Gaussian Mixture Model, allowing each document to belong to multiple clusters with probability distributions.

### Semantic Cache

The cache stores embeddings of previous queries. If a new query is semantically similar to a cached query, the cached results are returned without performing another vector search.

---

## Embedding Model Choice

The model **all-MiniLM-L6-v2** from the SentenceTransformers library was selected for generating embeddings.

Reasons for this choice:

* **Semantic Understanding:** The model produces embeddings that capture semantic relationships between sentences rather than simple keyword matching.
* **Efficiency:** It is a lightweight transformer model (~384 dimensional embeddings) that balances performance and speed.
* **Industry Usage:** The model is widely used for semantic search, clustering, and retrieval tasks.

Using embeddings allows the system to retrieve documents based on meaning rather than exact word overlap.

## Vector Database Choice

FAISS (Facebook AI Similarity Search) was used as the vector database for indexing and searching document embeddings.

Reasons for selecting FAISS:

* **High Performance:** FAISS provides extremely efficient nearest neighbor search for high-dimensional vectors.
* **Scalability:** It can scale to millions of vectors while maintaining fast query performance.
* **Flexibility:** The library provides multiple indexing strategies depending on the use case.

For this implementation, a simple **IndexFlatL2** index was used to perform exact similarity search over the embeddings.

## Fuzzy Clustering Approach

Instead of assigning each document to a single cluster, the system uses **fuzzy clustering** through a Gaussian Mixture Model (GMM).

In this approach:

* Each cluster is modeled as a probability distribution.
* A document receives a probability score for belonging to each cluster.
* Documents may therefore belong to multiple clusters with varying confidence.

For example:

Document A →
Cluster 1: 0.65
Cluster 2: 0.25
Cluster 3: 0.10

This approach is beneficial because documents often contain multiple topics and cannot always be represented by a single cluster.


## Semantic Cache Design

A semantic caching mechanism is implemented to reduce redundant computations.

Traditional caching relies on exact key matching, which does not work well for natural language queries. For example:

Query 1: "gun control debate"
Query 2: "firearm legislation discussion"

Although the wording differs, the semantic meaning is similar.

To address this, the system stores:

* Query embeddings
* Corresponding search results

When a new query arrives:

1. The query is converted into an embedding.
2. Cosine similarity is computed against cached query embeddings.
3. If similarity exceeds a predefined threshold, the cached result is returned.

This approach significantly reduces repeated vector searches for semantically similar queries.


## Persistence Strategy

To avoid recomputing embeddings on every server restart, the system stores:

* Document embeddings (`embeddings.npy`)
* Document texts (`documents.pkl`)

On startup, the system checks whether these files exist.

If they are available, the embeddings are loaded directly from disk.
Otherwise, embeddings are generated and stored for future runs.

This significantly reduces startup time for subsequent executions.

## System Architecture

```
User Query
     │
     ▼
SentenceTransformer Embedding
     │
     ▼
Semantic Cache Check
     │
     ├── Cache Hit → Return Cached Results
     │
     ▼
FAISS Vector Search
     │
     ▼
Retrieve Top Documents
     │
     ▼
Cluster Mapping (GMM)
     │
     ▼
Return API Response
```


## API Endpoints

### POST `/query`

Search for documents.

Example request:

```
{
 "query": "gun control debate"
}
```

Example response:

```
{
 "query": "gun control debate",
 "cache_hit": false,
 "results": [...],
 "clusters": [2, 3, 1]
}
```

---

### GET `/cache/stats`

Returns cache statistics.

---

### DELETE `/cache`

Clears the semantic cache.

---

## Setup

Install dependencies:

```
pip install -r requirements.txt
```

Run the API:

```
uvicorn main:app --reload
```

Open Swagger UI:

```
http://localhost:8000/docs
```

---

## Dataset

The system uses the **20 Newsgroups dataset**, a popular benchmark dataset containing approximately 20,000 newsgroup posts across multiple topics.

---

## Technologies Used

* Python
* FastAPI
* SentenceTransformers
* FAISS
* Scikit-Learn
* NumPy
