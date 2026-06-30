# src/agents/synthesizer.py
"""
Synthesis Agent: Bertugas membaca dokumen hasil retrieval 
dan menyusun draft jawaban menggunakan Gemini 2.5 Flash.
"""

import logging
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage
from src.agents.state import ScholarState

logger = logging.getLogger(__name__)

# Inisialisasi LLM Gemini
# temperature=0.2 digunakan agar jawabannya tetap faktual dan tidak terlalu berhalusinasi kreatif
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.2)

def synthesize_answer(state: ScholarState) -> ScholarState:
    query = state["query"]
    docs = state.get("documents", [])
    critique = state.get("critique", "")
    rev_count = state.get("revision_count", 0)

    logger.info(f"Synthesis Agent aktif. Memproses draft (Revisi ke-{rev_count})")

    # Gabungkan semua teks chunk menjadi satu string konteks
    context_str = "\n\n".join([
        f"--- SUMBER: {d['paper_title']} (Section: {d['section']}) ---\n{d['text']}" 
        for d in docs
    ])

    prompt = f"""Anda adalah asisten riset akademik profesional.
Tugas Anda adalah menjawab pertanyaan pengguna berdasarkan KONTEKS yang disediakan.
ATURAN MUTLAK:
1. Jika jawaban tidak ada di dalam teks konteks, katakan dengan jujur "Informasi tidak ditemukan dalam dokumen."
2. Sertakan kutipan sumber (Nama Paper) di akhir kalimat yang relevan.

PERTANYAAN: {query}

KONTEKS:
{context_str}
"""

    # Jika ada kritik dari putaran sebelumnya, tambahkan ke instruksi
    if critique:
        prompt += f"\n\nPERHATIAN! Draft Anda sebelumnya mendapat KRITIK berikut:\n{critique}\nPerbaiki draft Anda berdasarkan kritik di atas!"

    response = llm.invoke([HumanMessage(content=prompt)])
    
    # Perbarui state dengan draft baru dan tambah hitungan revisi
    return {
        "draft_answer": response.content,
        "revision_count": rev_count + 1
    }