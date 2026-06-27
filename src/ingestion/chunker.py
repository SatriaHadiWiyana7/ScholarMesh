# src/ingestion/chunker.py
"""
Strategi chunking untuk paper akademik:
1. Deteksi section heading menggunakan pola regex umum.
2. Chunk per section; jika section terlalu panjang (>max_tokens),
   pecah lagi dengan sliding window yang overlap.
"""

import re
from typing import List, Dict

SECTION_PATTERNS = re.compile(
    r"^(abstract|introduction|related work|background|method(ology)?|"
    r"experiment(s)?|result(s)?|discussion|conclusion(s)?|references)\b",
    re.IGNORECASE,
)

MAX_TOKENS_PER_CHUNK = 600    # perkiraan, ~1 token ≈ 0.75 kata
OVERLAP_TOKENS = 80            # overlap untuk menjaga kontinuitas konteks

def detect_sections(pages: List[Dict]) -> List[Dict]:
    """Gabungkan semua halaman jadi satu teks dengan deteksi section."""
    sections = []
    current_section = "preamble"
    current_text = []

    for page in pages:
        lines = page["text"].split("\n")
        for line in lines:
            stripped = line.strip()
            if SECTION_PATTERNS.match(stripped) and len(stripped) < 60:
                # Heading section baru terdeteksi, simpan section sebelumnya
                if current_text:
                    sections.append({
                        "section": current_section,
                        "text": "\n".join(current_text),
                    })
                current_section = stripped
                current_text = []
            else:
                current_text.append(line)

    if current_text:
        sections.append({"section": current_section, "text": "\n".join(current_text)})

    return sections

def sliding_window_split(text: str, max_tokens: int, overlap: int) -> List[str]:
    """Fallback: pecah teks panjang dengan sliding window berbasis kata."""
    words = text.split()
    if len(words) <= max_tokens:
        return [text]

    chunks = []
    step = max_tokens - overlap
    for start in range(0, len(words), step):
        chunk_words = words[start:start + max_tokens]
        chunks.append(" ".join(chunk_words))
        if start + max_tokens >= len(words):
            break
    return chunks

def chunk_paper(pages: List[Dict], metadata: Dict) -> List[Dict]:
    """
    Pipeline lengkap: deteksi section, lalu pecah section panjang
    dengan sliding window. Setiap chunk membawa metadata lengkap
    untuk keperluan sitasi.
    """
    sections = detect_sections(pages)
    final_chunks = []

    for sec in sections:
        sub_chunks = sliding_window_split(
            sec["text"], MAX_TOKENS_PER_CHUNK, OVERLAP_TOKENS
        )
        for sub in sub_chunks:
            if len(sub.strip()) < 30:  # skip chunk kosong/terlalu pendek
                continue
            final_chunks.append({
                "text": sub.strip(),
                "section": sec["section"],
                "paper_title": metadata["title"],
                "year": metadata.get("year"),
                "authors": metadata.get("authors", "unknown"),
                "page": None,  # opsional: bisa diisi jika tracking per-halaman dibutuhkan
            })

    return final_chunks