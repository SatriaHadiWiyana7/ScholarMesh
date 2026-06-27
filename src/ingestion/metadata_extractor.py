# src/ingestion/metadata_extractor.py
"""
Ekstraksi metadata paper (judul, penulis, tahun) dari halaman pertama.
Menggunakan heuristik font-size, dengan fallback LLM untuk kasus ambigu.
"""

import fitz
from pathlib import Path
from typing import Dict
import re

def extract_metadata_heuristic(pdf_path: Path) -> Dict:
    doc = fitz.open(pdf_path)
    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]

    max_font_size = 0
    title_candidate = ""

    for block in blocks:
        if "lines" not in block:
            continue
        for line in block["lines"]:
            for span in line["spans"]:
                if span["size"] > max_font_size and len(span["text"].strip()) > 10:
                    max_font_size = span["size"]
                    title_candidate = span["text"].strip()

    # Cari tahun dengan regex sederhana (4 digit, 1990-2029)
    full_text = first_page.get_text()
    year_match = re.search(r"\b(19[9]\d|20[0-2]\d)\b", full_text)
    year = int(year_match.group()) if year_match else None

    doc.close()
    return {
        "title": title_candidate or pdf_path.stem,
        "year": year,
        "source_file": pdf_path.name,
    }