# src/retrieval/qdrant_client_setup.py
"""
Setup koneksi dan skema collection Qdrant untuk ScholarMesh.
Collection mendukung hybrid search: named dense vector + named sparse vector.
"""

from qdrant_client import QdrantClient
from qdrant_client.models import (
    VectorParams, Distance, SparseVectorParams,
    SparseIndexParams, PointStruct, SparseVector
)
import uuid
import logging

logger = logging.getLogger(__name__)

COLLECTION_NAME = "scholarmesh_papers"
DENSE_DIM = 1024  # sesuai dimensi output BGE-M3

def get_client(host: str = "localhost", port: int = 6333) -> QdrantClient:
    return QdrantClient(host=host, port=port)

def create_collection_if_not_exists(client: QdrantClient) -> None:
    existing = [c.name for c in client.get_collections().collections]
    if COLLECTION_NAME in existing:
        logger.info(f"Collection '{COLLECTION_NAME}' sudah ada, skip pembuatan.")
        return

    client.create_collection(
        collection_name=COLLECTION_NAME,
        vectors_config={
            "dense": VectorParams(size=DENSE_DIM, distance=Distance.COSINE),
        },
        sparse_vectors_config={
            "sparse": SparseVectorParams(index=SparseIndexParams(on_disk=False)),
        },
    )
    logger.info(f"Collection '{COLLECTION_NAME}' berhasil dibuat.")

def upsert_chunks(client: QdrantClient, chunks: list[dict], embeddings: dict) -> None:
    """
    chunks: list of {"text": str, "paper_title": str, "section": str,
                       "page": int, "year": int, "authors": str}
    embeddings: output dari BGEM3Encoder.encode_documents()
    """
    points = []
    dense_vecs = embeddings["dense_vecs"]
    lexical_weights = embeddings["lexical_weights"]

    for i, chunk in enumerate(chunks):
        sparse = lexical_weights[i]
        indices = list(sparse.keys())
        values = list(sparse.values())

        points.append(
            PointStruct(
                id=str(uuid.uuid4()),
                vector={
                    "dense": dense_vecs[i].tolist(),
                    "sparse": SparseVector(indices=indices, values=values),
                },
                payload={
                    "text": chunk["text"],
                    "paper_title": chunk["paper_title"],
                    "section": chunk["section"],
                    "page": chunk.get("page"),
                    "year": chunk.get("year"),
                    "authors": chunk.get("authors"),
                },
            )
        )

    client.upsert(collection_name=COLLECTION_NAME, points=points)