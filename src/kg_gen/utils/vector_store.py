import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from kg_gen.utils.chunk_text import chunk_text


@dataclass(frozen=True)
class VectorIndexMetadata:
    embedding_model: str
    chunk_size: int
    chunk_count: int
    created_at: str


class VectorStore:
    def __init__(self, base_dir: Path):
        self.base_dir = Path(base_dir)
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def _graph_dir(self, graph_id: str) -> Path:
        graph_dir = self.base_dir / graph_id
        graph_dir.mkdir(parents=True, exist_ok=True)
        return graph_dir

    def _embeddings_path(self, graph_id: str) -> Path:
        return self._graph_dir(graph_id) / "embeddings.npy"

    def _chunks_path(self, graph_id: str) -> Path:
        return self._graph_dir(graph_id) / "chunks.json"

    def _meta_path(self, graph_id: str) -> Path:
        return self._graph_dir(graph_id) / "index.json"

    def build_index(
        self,
        graph_id: str,
        text: str,
        embedding_model: str,
        chunk_size: int,
    ) -> VectorIndexMetadata:
        cleaned_text = text.strip()
        if not cleaned_text:
            raise ValueError("Text is required to build a vector index")

        chunks = chunk_text(cleaned_text, max_chunk_size=chunk_size)
        if not chunks:
            raise ValueError("No chunks generated from provided text")

        model = SentenceTransformer(embedding_model)
        embeddings = model.encode(chunks, show_progress_bar=False)

        embeddings_path = self._embeddings_path(graph_id)
        chunks_path = self._chunks_path(graph_id)
        meta_path = self._meta_path(graph_id)

        np.save(embeddings_path, embeddings)

        chunk_payload = [
            {"id": idx, "text": chunk, "length": len(chunk)}
            for idx, chunk in enumerate(chunks)
        ]
        with open(chunks_path, "w", encoding="utf-8") as f:
            json.dump(chunk_payload, f, indent=2, ensure_ascii=False)

        meta = VectorIndexMetadata(
            embedding_model=embedding_model,
            chunk_size=chunk_size,
            chunk_count=len(chunks),
            created_at=datetime.now().isoformat(),
        )
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta.__dict__, f, indent=2, ensure_ascii=False)

        return meta

    def load_index(self, graph_id: str):
        embeddings_path = self._embeddings_path(graph_id)
        chunks_path = self._chunks_path(graph_id)
        meta_path = self._meta_path(graph_id)

        if not embeddings_path.exists() or not chunks_path.exists() or not meta_path.exists():
            return None

        embeddings = np.load(embeddings_path)
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)

        return embeddings, chunks, meta

    def query(
        self,
        graph_id: str,
        query: str,
        top_k: int,
        embedding_model: Optional[str] = None,
    ) -> list[dict]:
        loaded = self.load_index(graph_id)
        if loaded is None:
            raise ValueError("Vector index not found")

        embeddings, chunks, meta = loaded
        model_name = embedding_model or meta.get("embedding_model")
        if not model_name:
            raise ValueError("Embedding model is required to query the vector index")
        if embedding_model and meta.get("embedding_model") and embedding_model != meta.get("embedding_model"):
            raise ValueError("Embedding model does not match stored vector index")

        model = SentenceTransformer(model_name)
        query_embedding = model.encode([query], show_progress_bar=False)
        scores = cosine_similarity(query_embedding, embeddings).flatten()

        top_k = min(max(top_k, 1), len(chunks))
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            chunk = chunks[int(idx)]
            results.append({
                "id": chunk.get("id", int(idx)),
                "text": chunk.get("text", ""),
                "score": float(scores[int(idx)]),
                "length": chunk.get("length"),
            })

        return results
