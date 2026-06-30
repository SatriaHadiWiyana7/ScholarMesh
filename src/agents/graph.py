# src/agents/graph.py
"""
Mendefinisikan StateGraph (Sirkuit LangGraph) untuk sistem Multi-Agent.
"""

from langgraph.graph import StateGraph, END
from src.agents.state import ScholarState
from src.agents.retriever import retrieve_documents
from src.agents.synthesizer import synthesize_answer
from src.agents.critique import evaluate_draft

def route_evaluation(state: ScholarState):
    """Fungsi router untuk menentukan apakah loop selesai atau harus revisi."""
    if not state.get("critique"):
        return END
    
    # Mencegah infinite loop (Agent berdebat terus-menerus tanpa henti)
    if state.get("revision_count", 0) >= 3:
        return END
        
    return "synthesize"

# Inisialisasi Graph
workflow = StateGraph(ScholarState)

# Daftarkan Node (Agent Anda)
workflow.add_node("retrieve", retrieve_documents)
workflow.add_node("synthesize", synthesize_answer)
workflow.add_node("evaluator", evaluate_draft)

# Buat Alur Maju (Edges)
workflow.set_entry_point("retrieve")
workflow.add_edge("retrieve", "synthesize")
workflow.add_edge("synthesize", "evaluator")

# Conditional Edge dari Evaluator (Penentu Keputusan)
workflow.add_conditional_edges(
    "evaluator",
    route_evaluation,
    {
        "synthesize": "synthesize",
        END: END
    }
)

# 5. Compile menjadi aplikasi (Runnable)
app = workflow.compile()