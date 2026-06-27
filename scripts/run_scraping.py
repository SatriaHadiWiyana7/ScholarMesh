# scripts/run_scraping.py
"""
CLI untuk menjalankan proses scraping paper dari arXiv.
Jalankan dari root proyek: python scripts/run_scraping.py
"""

import logging
from pathlib import Path
from src.scraping.arxiv_client import fetch_and_download_papers

# Setup konfigurasi logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

# path untuk menyimpan paper yang diunduh
RAW_PAPERS_DIR = Path("data/raw_papers")

# Konfigurasi pencarian
SEARCH_QUERY = "Large Language Models Multi-Agent" 
MAX_PAPERS = 15

def main():
    fetch_and_download_papers(
        query=SEARCH_QUERY,
        max_results=MAX_PAPERS,
        output_dir=RAW_PAPERS_DIR
    )

if __name__ == "__main__":
    main()