# src/ingestion/pdf_parser.py
"""
Parser PDF akademik menggunakan PyMuPDF.
Menangani layout dua kolom dan ekstraksi terstruktur per halaman.
"""

import fitz  # PyMuPDF
from pathlib import Path
from typing import List, Dict
import logging

logger = logging.getLogger(__name__)

def is_likely_scanned(doc: fitz.Document, sample_pages: int = 3) -> bool:
    """Deteksi apakah PDF adalah hasil scan (teks tidak dapat diekstrak)."""
    pages_to_check = min(sample_pages, len(doc))
    total_text_len = 0
    for i in range(pages_to_check):
        total_text_len += len(doc[i].get_text().strip())
    # Jika hampir tidak ada teks yang terekstrak, kemungkinan hasil scan
    return total_text_len < 50

def extract_text_with_layout(pdf_path: Path) -> List[Dict]:
    """
    Ekstrak teks per halaman, mempertahankan urutan baca yang benar
    untuk layout dua kolom (PyMuPDF menangani ini secara native
    lewat parameter sort=True pada get_text("blocks")).
    """
    doc = fitz.open(pdf_path)

    if is_likely_scanned(doc):
        logger.warning(f"{pdf_path.name} terdeteksi sebagai scan, perlu OCR.")
        doc.close()
        from src.ingestion.ocr_fallback import extract_text_via_ocr
        return extract_text_via_ocr(pdf_path)

    pages = []
    for page_num, page in enumerate(doc):
        blocks = page.get_text("blocks", sort=True)  # sort=True: urutan baca benar
        page_text = "\n".join(b[4] for b in blocks if b[6] == 0)  # type 0 = teks
        pages.append({
            "page_number": page_num + 1,
            "text": page_text,
        })

    doc.close()
    return pages