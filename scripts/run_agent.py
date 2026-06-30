# scripts/run_agent.py
"""
Script utama untuk menguji interaksi langsung dengan Multi-Agent.
Jalankan dari root folder: PYTHONPATH=. python scripts/run_agent.py
"""

import logging
from dotenv import load_dotenv

# WAJIB diletakkan sebelum import graph agar API Key termuat lebih dulu
load_dotenv() 

from src.agents.graph import app

# Supaya Anda bisa melihat diskusi antar Agent secara real-time
logging.basicConfig(level=logging.INFO, format='%(name)s - %(message)s')

def main():
    print("\n" + "="*50)
    print("🤖 Selamat datang di ScholarMesh Multi-Agent!")
    print("Ketik 'keluar' untuk berhenti.")
    print("="*50 + "\n")
    
    while True:
        user_query = input("\nMasukkan pertanyaan Anda (seputar paper): ")
        if user_query.lower() in ['keluar', 'exit', 'quit']:
            print("Sampai jumpa!")
            break
            
        initial_state = {
            "query": user_query,
            "revision_count": 0,
            "critique": ""
        }
        
        print("\n⏳ Agent sedang bekerja membedah dokumen...")
        
        result = app.invoke(initial_state)
        
        print("\n" + "="*50)
        print("✅ JAWABAN FINAL:")
        print("="*50)
        final_ans = result.get("final_answer")
        if not final_ans:
            final_ans = result.get("draft_answer", "Maaf, sistem gagal menghasilkan jawaban.")
            
        print(final_ans)
        print("="*50)

if __name__ == "__main__":
    main()