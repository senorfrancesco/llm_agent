from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any
import json
import uvicorn

# ============================================================================
# 1. Document Server Mock
# ============================================================================

app_doc = FastAPI(title="MCP Document Server Mock")

class LoadDocumentRequest(BaseModel):
    path: str

@app_doc.post("/load_document")
async def load_document(request: LoadDocumentRequest):
    """Имитация загрузки документа и возврата текста."""
    print(f"[DOC_SERVER] Received request to load: {request.path}")
    
    # Имитация логики, которая в реальном проекте вызывает UMS для OCR/парсинга
    if "contract_old" in request.path:
        text = "Document A: Contract text (old version). Clause 1: Price is $100. Clause 2: Delivery in 30 days."
    elif "contract_new" in request.path:
        text = "Document B: Contract text (new version). Clause 1: Price is $120. Clause 2: Delivery in 15 days."
    else:
        text = f"Document {request.path} loaded successfully."
        
    return {"status": "success", "text": text}

# ============================================================================
# 2. Legal Server Mock
# ============================================================================

app_legal = FastAPI(title="MCP Legal Server Mock")

class CompareChunksRequest(BaseModel):
    old_text: str
    new_text: str

class AnalyzeImpactRequest(BaseModel):
    differences: str

@app_legal.post("/compare_chunks")
async def compare_chunks(request: CompareChunksRequest):
    """Имитация сравнения двух текстов."""
    print("[LEGAL_SERVER] Received request to compare chunks.")
    
    # Имитация логики, которая в реальном проекте вызывает UMS для эмбеддингов
    diff = []
    if "100" in request.old_text and "120" in request.new_text:
        diff.append({"type": "MODIFIED", "detail": "Price changed from $100 to $120."})
    if "30 days" in request.old_text and "15 days" in request.new_text:
        diff.append({"type": "MODIFIED", "detail": "Delivery time reduced from 30 to 15 days."})
        
    return {"status": "success", "differences": diff}

@app_legal.post("/analyze_impact")
async def analyze_impact(request: AnalyzeImpactRequest):
    """Имитация анализа юридической значимости."""
    print("[LEGAL_SERVER] Received request to analyze impact.")
    
    # Имитация логики, которая в реальном проекте вызывает UMS для LLM-анализа
    if "Price changed" in request.differences:
        impact = "The price change is a critical financial impact. The delivery time reduction is a positive operational change."
    else:
        impact = "No critical impact found."
        
    return {"status": "success", "impact": impact}

# ============================================================================
# 3. Запуск серверов (для тестирования)
# ============================================================================

if __name__ == "__main__":
    # Запуск Document Server на порту 8001
    uvicorn.run(app_doc, host="0.0.0.0", port=8001)
    
    # Запуск Legal Server на порту 8002
    # В реальном проекте нужно запускать в разных процессах
    # uvicorn.run(app_legal, host="0.0.0.0", port=8002)
    
    # Для простоты, в прототипе мы будем использовать функции-заглушки
    # в react_agent.py, которые имитируют HTTP-вызовы.
    print("MCP Server Mocks defined. Run react_agent.py to see the ReAct logic.")
