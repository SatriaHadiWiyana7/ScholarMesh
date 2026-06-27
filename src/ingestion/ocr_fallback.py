# src/ingestion/ocr_fallback.py
"""
Fallback OCR menggunakan Tesseract untuk PDF yang merupakan hasil scan
(teks tidak dapat diekstrak langsung).
"""

import fitz
import pytesseract
from PIL import Image
import io
from pathlib import Path
from typing import List, Dict

def extract_text_via_ocr(pdf_path: Path, dpi: int = 200) -> List[Dict]:
    doc = fitz.open(pdf_path)
    pages = []

    for page_num, page in enumerate(doc):
        pix = page.get_pixmap(dpi=dpi)
        img = Image.open(io.BytesIO(pix.tobytes("png")))
        text = pytesseract.image_to_string(img, lang="eng+ind")
        pages.append({
            "page_number": page_num + 1,
            "text": text,
        })

    doc.close()
    return pages