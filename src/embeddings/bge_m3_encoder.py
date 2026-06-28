# src/embeddings/bge_m3_encoder.py
"""
Wrapper untuk BGE-M3 embedding model.
Menghasilkan dense vector dan sparse (lexical) weights dalam satu panggilan.
"""

from FlagEmbedding import BGEM3FlagModel
from typing import List, Dict, Any
import numpy as np
import logging

logger = logging.getLogger(__name__)

class BGEM3Encoder:
    """
    Wrapper BGE-M3 untuk keperluan hybrid retrieval ScholarMesh.

    Args:
        model_name: nama model di Hugging Face Hub.
        use_fp16: gunakan precision fp16 untuk mempercepat inference
                   (aman di CPU modern & GPU; non-aktifkan jika hasil aneh).
        device: 'cpu' atau 'cuda' (auto-detect jika None).
    """

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        use_fp16: bool = True,
        device: str | None = None,
    ):
        logger.info(f"Memuat model embedding: {model_name} (fp16={use_fp16})")
        self.model = BGEM3FlagModel(
            model_name,
            use_fp16=use_fp16,
            device=device,
        )

    def encode_documents(
        self,
        texts: List[str],
        batch_size: int = 12,
        max_length: int = 8192,
    ) -> Dict[str, Any]:
        """
        Encode batch dokumen/chunk menjadi dense + sparse representation.

        Returns:
            {
                "dense_vecs": np.ndarray shape (N, 1024),
                "lexical_weights": List[Dict[token_id, weight]],
            }
        """
        output = self.model.encode(
            texts,
            batch_size=batch_size,
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,  # multi-vector di-nonaktifkan dulu
        )
        return {
            "dense_vecs": output["dense_vecs"],
            "lexical_weights": output["lexical_weights"],
        }

    def encode_query(self, query: str, max_length: int = 512) -> Dict[str, Any]:
        """
        Encode satu query user. max_length lebih kecil dari dokumen,
        karena query akademik biasanya singkat dan padat.
        """
        output = self.model.encode(
            [query],
            max_length=max_length,
            return_dense=True,
            return_sparse=True,
            return_colbert_vecs=False,
        )
        return {
            "dense_vec": output["dense_vecs"][0],
            "lexical_weights": output["lexical_weights"][0],
        }

    @staticmethod
    def dense_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
        """Cosine similarity antar dua dense vector."""
        return float(
            np.dot(vec_a, vec_b) / (np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
        )