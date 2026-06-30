
# src/agents/state.py
"""
Definisi State/Memori Bersama untuk LangGraph.
Setiap node (Agent) akan membaca dan memperbarui dictionary ini.
"""

from typing import TypedDict, List, Dict, Any

class ScholarState(TypedDict):
    query: str                  # Pertanyaan asli dari user
    documents: List[Dict[str, Any]]  # Chunk teks yang diambil dari Qdrant
    draft_answer: str           # Jawaban sementara dari Synthesis Agent
    critique: str               # Kritik/masukan dari Critique Agent
    revision_count: int         # Menghitung jumlah revisi agar tidak looping tak terbatas
    final_answer: str           # Jawaban akhir yang disetujui