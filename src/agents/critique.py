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

# Inisialisasi LLM
llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0.0)

_PROMPT_TEMPLATE = """Anda adalah evaluator ketat untuk sistem RAG akademik.
Tugas Anda: evaluasi DRAFT JAWABAN menggunakan KONTEKS ASLI sebagai acuan kebenaran.

═══════════════════════════════════════
PERTANYAAN PENGGUNA:
{query}

DRAFT JAWABAN:
{draft}

KONTEKS ASLI (sumber kebenaran):
{context}
═══════════════════════════════════════

KRITERIA EVALUASI (periksa satu per satu):
1. RELEVANSI   — Apakah draft menjawab pertanyaan secara langsung?
2. FAKTUALITAS — Apakah setiap klaim dalam draft didukung oleh konteks?
               Tandai setiap klaim yang TIDAK ada di konteks sebagai HALUSINASI.
3. KELENGKAPAN — Apakah ada informasi penting di konteks yang terlewat?
4. SITASI       — Apakah setiap klaim faktual sudah disertai nama paper/section?

Tuliskan evaluasi Anda secara ringkas per kriteria.

KEMUDIAN, di baris PALING AKHIR tuliskan PERSIS salah satu dari dua opsi ini:
PUTUSAN: TERIMA
PUTUSAN: TOLAK
"""


def _parse_verdict(evaluation_text: str) -> str:
    """
    Cari putusan HANYA di 3 baris terakhir output Critique Agent.

    Mengapa tidak mencari di seluruh teks?
    LLM sering menulis kalimat seperti "klaim ini perlu ditolak" atau
    "saya menolak argumen X" di dalam teks evaluasi, yang bisa men-
    trigger false-positive jika kita scan seluruh string.

    Return: "TERIMA" | "TOLAK" | "UNKNOWN"
    """
    last_lines = evaluation_text.strip().split("\n")[-3:]
    tail = " ".join(last_lines).upper()

    if "PUTUSAN: TERIMA" in tail:
        return "TERIMA"
    if "PUTUSAN: TOLAK" in tail:
        return "TOLAK"
    return "UNKNOWN"


def evaluate_draft(state: ScholarState) -> dict:
    """
    Node function untuk LangGraph.

    Alur:
    1. Guard: draft kosong → TOLAK otomatis.
    2. Bangun prompt evaluasi dengan konteks + draft.
    3. Parse putusan dari 3 baris terakhir output LLM.
    4. TERIMA  → simpan draft sebagai final_answer, kosongkan critique.
       TOLAK   → simpan teks evaluasi sebagai critique, kosongkan final_answer.
       UNKNOWN → default TERIMA + log warning (mencegah infinite loop).
    """
    query  = state["query"]
    docs   = state.get("documents") or []
    draft  = state.get("draft_answer") or ""

    logger.info("Critique Agent aktif. Mengevaluasi draft...")

    # Guard: draft kosong
    if not draft:
        logger.warning("Critique Agent menerima draft kosong dari Synthesizer.")
        return {
            "critique": (
                "Draft jawaban kosong. Synthesizer gagal menghasilkan respons. "
                "Coba susun ulang jawaban dari konteks yang tersedia."
            ),
            "final_answer": None,
        }

    # Bangun konteks
    context_str = "\n\n".join(
        f"[Paper: {d['paper_title']} | Section: {d['section']}]\n{d['text']}"
        for d in docs
    )

    prompt = _PROMPT_TEMPLATE.format(
        query=query,
        draft=draft,
        context=context_str or "(tidak ada konteks — retrieval kosong)",
    )

    # Panggil Gemini
    try:
        response = llm.invoke([HumanMessage(content=prompt)])
        evaluation_text = response.content.strip()
    except Exception as e:
        logger.error(f"Gemini API error di Critique Agent: {e}")
        return {
            "final_answer": draft,
            "critique": "",
        }

    # Parse putusan
    verdict = _parse_verdict(evaluation_text)

    if verdict == "TERIMA":
        logger.info("Draft DITERIMA oleh Critique Agent.")
        return {
            "final_answer": draft,
            "critique": "",
        }

    if verdict == "TOLAK":
        logger.warning("Draft DITOLAK. Mengembalikan ke Synthesizer dengan critique.")
        return {
            "critique": evaluation_text,
            "final_answer": None,    # belum ada jawaban final
        }

    # format putusan tidak ditemukan
    logger.warning(
        "Critique Agent tidak mengikuti format PUTUSAN. "
        "Default ke TERIMA untuk mencegah infinite loop.\n"
        f"Output (100 char terakhir): ...{evaluation_text[-100:]}"
    )
    return {
        "final_answer": draft,
        "critique": "",
    }