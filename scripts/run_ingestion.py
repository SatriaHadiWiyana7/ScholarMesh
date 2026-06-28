# scripts/run_ingestion.py
"""
CLI untuk menjalankan seluruh pipeline ingestion:
PDF mentah -> parsing -> chunking -> embedding -> Qdrant.
"""

from pathlib import Path
from tqdm import tqdm
import logging

from src.ingestion.pdf_parser import extract_text_with_layout
from src.ingestion.metadata_extractor import extract_metadata_heuristic
from src.ingestion.chunker import chunk_paper
from src.embeddings.bge_m3_encoder import BGEM3Encoder
from src.retrieval.qdrant_client_setup import (
    get_client, create_collection_if_not_exists, upsert_chunks,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

RAW_PAPERS_DIR = Path("data/raw_papers")

def main():
    client = get_client()
    create_collection_if_not_exists(client)
    encoder = BGEM3Encoder()

    pdf_files = list(RAW_PAPERS_DIR.glob("*.pdf"))
    logger.info(f"Ditemukan {len(pdf_files)} paper untuk diproses.")

    for pdf_path in tqdm(pdf_files, desc="Memproses paper"):
        try:
            pages = extract_text_with_layout(pdf_path)
            metadata = extract_metadata_heuristic(pdf_path)
            chunks = chunk_paper(pages, metadata)

            if not chunks:
                logger.warning(f"Tidak ada chunk dihasilkan untuk {pdf_path.name}, skip.")
                continue

            texts = [c["text"] for c in chunks]
            embeddings = encoder.encode_documents(texts)
            upsert_chunks(client, chunks, embeddings)

            logger.info(f"{pdf_path.name}: {len(chunks)} chunk berhasil diunggah.")
        except Exception as e:
            logger.error(f"Gagal memproses {pdf_path.name}: {e}")
            continue

    logger.info("Ingestion selesai.")

if __name__ == "__main__":
    main()