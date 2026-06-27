# src/scraping/arxiv_client.py
"""
Modul klien untuk berinteraksi dengan API arXiv dan mengunduh file PDF.
"""

import arxiv
import urllib.request
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def clean_filename(title: str) -> str:
    """Membersihkan judul agar aman dijadikan nama file."""
    safe_title = "".join([c for c in title if c.isalpha() or c.isdigit() or c == ' ']).rstrip()
    return safe_title.replace(" ", "_")

def fetch_and_download_papers(query: str, max_results: int, output_dir: Path) -> None:
    """
    Mencari paper di arXiv berdasarkan query dan mengunduhnya ke direktori output.
    """
    logger.info(f"Memulai pencarian arXiv untuk query: '{query}' (Maksimal: {max_results} paper)")
    
    # Pastikan direktori output sudah ada
    output_dir.mkdir(parents=True, exist_ok=True)
    
    search = arxiv.Search(
        query=query,
        max_results=max_results,
        sort_by=arxiv.SortCriterion.Relevance
    )
    
    client = arxiv.Client()
    downloaded_count = 0
    
    for paper in client.results(search):
        filename = f"{clean_filename(paper.title)}.pdf"
        filepath = output_dir / filename
        
        if filepath.exists():
            logger.info(f"File sudah ada, melewati: {filename}")
            continue
            
        logger.info(f"Mengunduh [{downloaded_count + 1}/{max_results}]: {filename}")
        
        try:
            # Unduh PDF langsung dari URL yang disediakan arXiv
            urllib.request.urlretrieve(paper.pdf_url, filepath)
            downloaded_count += 1
        except Exception as e:
            logger.error(f"Gagal mengunduh '{filename}': {e}")
            
    logger.info(f"Proses scraping selesai. Berhasil mengunduh {downloaded_count} paper baru.")