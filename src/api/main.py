# src/api/main.py
"""
Backend FastAPI untuk ScholarMesh.
Menghubungkan React Frontend dengan LangGraph Multi-Agent.
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

# Muat API Key sebelum memanggil Agent
load_dotenv()

from src.agents.graph import app as agent_app

app = FastAPI(title="ScholarMesh AI API")

# Konfigurasi CORS agar React (yang berjalan di port berbeda) bisa mengakses API ini
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Untuk production, ganti dengan URL React Anda
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Definisi struktur data yang diterima dari Frontend
class ChatRequest(BaseModel):
    query: str

@app.post("/api/chat")
async def process_chat(request: ChatRequest):
    try:
        initial_state = {
            "query": request.query,
            "revision_count": 0,
            "critique": ""
        }
        
        # Eksekusi StateGraph
        result = agent_app.invoke(initial_state)
        
        # Ambil jawaban akhir atau draft jika gagal
        final_ans = result.get("final_answer")
        if not final_ans:
            final_ans = result.get("draft_answer", "Maaf, sistem gagal menghasilkan jawaban.")
            
        # Kembalikan data lengkap
        return {
            "answer": final_ans,
            "critique": result.get("critique", ""),
            "sources": result.get("documents", [])
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True, reload_dirs=["src"])