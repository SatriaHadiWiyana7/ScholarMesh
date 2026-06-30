# src/agents/retriever.py
"""
Retrieval Agent: Bertugas mengambil chunk relevan dari Qdrant
berdasarkan query pengguna dan memperbarui state.
"""

import logging
from src.agents.state import ScholarState
from src.embeddings.bge_m3_encoder import BGEM3Encoder
from src.retrieval.qdrant_client_setup import get_client, COLLECTION_NAME

logger = logging.getLogger(__name__)

# Inisialisasi model dan client di luar fungsi agar tidak dimuat berulang kali
encoder = BGEM3Encoder()
qdrant_client = get_client()

def retrieve_documents(state: ScholarState) -> ScholarState:
    """
    Node function untuk LangGraph:
    Mengambil query, mencari 5 chunk paling relevan di Qdrant, 
    dan menyimpannya ke dalam state.
    """
    query = state["query"]
    logger.info(f"Retrieval Agent aktif. Mencari dokumen untuk query: '{query}'")
    
    # Ubah query menjadi vektor
    encoded_query = encoder.encode_query(query)
    dense_vector = encoded_query["dense_vec"].tolist()
    
    # Cari di Qdrant
    search_results = qdrant_client.search(
        collection_name=COLLECTION_NAME,
        query_vector=("dense", dense_vector),
        limit=5,  # Ambil 5 chunk paling mirip
        with_payload=True
    )
    
    # Ekstrak teks dan metadata dari hasil pencarian
    retrieved_docs = []
    for hit in search_results:
        payload = hit.payload
        retrieved_docs.append({
            "text": payload.get("text", ""),
            "paper_title": payload.get("paper_title", "Unknown"),
            "section": payload.get("section", "Unknown"),
            "score": hit.score
        })
        
    logger.info(f"Berhasil menemukan {len(retrieved_docs)} dokumen relevan.")
    
    # 4. Perbarui state
    return {"documents": retrieved_docs}