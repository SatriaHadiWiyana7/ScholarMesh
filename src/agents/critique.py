# src/agents/critique.py
"""
Critique Agent: Mengevaluasi draft jawaban. Jika bagus, status diterima.
Jika ada halusinasi atau kurang lengkap, berikan kritik untuk direvisi.
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from src.agents.state import ScholarState

logger = logging.getLogger(__name__)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

def evaluate_draft(state: ScholarState) -> ScholarState:
    query = state["query"]
    docs = state.get("documents", [])
    draft = state.get("draft_answer", "")

    logger.info("Critique Agent aktif. Mengevaluasi kualitas draft...")

    context_str = "\n\n".join([d["text"] for d in docs])

    prompt = f"""Anda adalah evaluator ketat untuk sistem AI akademik.
Tugas Anda adalah mengevaluasi DRAFT JAWABAN berdasarkan KONTEKS ASLI.

PERTANYAAN: {query}
DRAFT JAWABAN: {draft}

KONTEKS ASLI:
{context_str}

EVALUASI:
1. Apakah draft menjawab pertanyaan?
2. Apakah draft mengandung halusinasi (fakta yang tidak ada di konteks)?

Tuliskan evaluasi Anda. Di baris paling akhir, Anda WAJIB memberikan putusan final dengan format persis seperti ini:
PUTUSAN: [TERIMA/TOLAK]
"""

    response = llm.invoke([HumanMessage(content=prompt)])
    evaluation_text = response.content.strip()
    
    # Analisis putusan
    if "PUTUSAN: TERIMA" in evaluation_text.upper():
        logger.info("Draft DITERIMA. Proses selesai.")
        return {"final_answer": draft, "critique": ""}
    else:
        logger.warning("Draft DITOLAK. Mengembalikan ke Synthesizer.")
        return {"critique": evaluation_text}