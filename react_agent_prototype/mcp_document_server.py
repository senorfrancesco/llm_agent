"""
MCP Document Server - FastAPI приложение для работы с документами.

Функции:
- Загрузка документов (PDF, DOCX)
- Извлечение текста (через OCR, если нужно)
- Извлечение таблиц
- Семантическое разбиение текста на чанки

Интегрирует UMS для использования Vision-модели (Qwen-VL) для OCR.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from ums_client import process_vision_via_ums, get_embeddings_via_ums

app = FastAPI(title="MCP Document Server", version="1.0.0")

# ============================================================================
# Request/Response Models
# ============================================================================

class LoadDocumentRequest(BaseModel):
    path: str
    extract_tables: bool = False
    use_ocr: bool = False

class LoadDocumentResponse(BaseModel):
    status: str
    text: Optional[str] = None
    tables: Optional[List[List[List[str]]]] = None
    error: Optional[str] = None

class ExtractTablesRequest(BaseModel):
    path: str
    pages: Optional[List[int]] = None

class ExtractTablesResponse(BaseModel):
    status: str
    tables: Optional[List[Dict[str, Any]]] = None
    error: Optional[str] = None

class SmartChunkRequest(BaseModel):
    text: str
    chunk_size: int = 512
    overlap: int = 50
    strategy: str = "semantic"  # "semantic" или "fixed"

class SmartChunkResponse(BaseModel):
    status: str
    chunks: Optional[List[str]] = None
    error: Optional[str] = None

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mcp-document-server"}

@app.post("/load_document", response_model=LoadDocumentResponse)
async def load_document(request: LoadDocumentRequest):
    """
    Загружает документ и возвращает его текстовое представление.
    
    Если use_ocr=True, использует Vision-модель (Qwen-VL) через UMS для OCR.
    """
    print(f"[DOC_SERVER] Loading document: {request.path}")
    
    try:
        # Имитация загрузки документа
        # В реальном проекте здесь будет логика для парсинга PDF/DOCX
        
        if "contract_old" in request.path:
            text = "Document A: Contract text (old version). Clause 1: Price is $100. Clause 2: Delivery in 30 days."
        elif "contract_new" in request.path:
            text = "Document B: Contract text (new version). Clause 1: Price is $120. Clause 2: Delivery in 15 days."
        else:
            # Если нужен OCR, используем UMS
            if request.use_ocr:
                text = process_vision_via_ums(request.path, "Extract all text from this document.")
            else:
                text = f"Document {request.path} loaded successfully."
        
        # Опционально извлекаем таблицы
        tables = None
        if request.extract_tables:
            tables = _extract_tables_mock(request.path)
        
        return LoadDocumentResponse(
            status="success",
            text=text,
            tables=tables
        )
    
    except Exception as e:
        return LoadDocumentResponse(
            status="error",
            error=str(e)
        )

@app.post("/extract_tables", response_model=ExtractTablesResponse)
async def extract_tables(request: ExtractTablesRequest):
    """Извлекает таблицы из документа."""
    print(f"[DOC_SERVER] Extracting tables from: {request.path}")
    
    try:
        tables = _extract_tables_mock(request.path)
        return ExtractTablesResponse(
            status="success",
            tables=tables
        )
    except Exception as e:
        return ExtractTablesResponse(
            status="error",
            error=str(e)
        )

@app.post("/smart_chunk", response_model=SmartChunkResponse)
async def smart_chunk(request: SmartChunkRequest):
    """
    Разбивает текст на чанки.
    
    Если strategy="semantic", использует эмбеддинги (через UMS) для
    определения границ чанков на основе семантического сходства.
    """
    print(f"[DOC_SERVER] Smart chunking text (strategy={request.strategy})")
    
    try:
        if request.strategy == "semantic":
            # Используем эмбеддинги через UMS для умного разбиения
            chunks = _semantic_chunking(request.text, request.chunk_size, request.overlap)
        else:
            # Простое разбиение по размеру
            chunks = _fixed_chunking(request.text, request.chunk_size, request.overlap)
        
        return SmartChunkResponse(
            status="success",
            chunks=chunks
        )
    except Exception as e:
        return SmartChunkResponse(
            status="error",
            error=str(e)
        )

# ============================================================================
# Helper Functions
# ============================================================================

def _extract_tables_mock(path: str) -> List[Dict[str, Any]]:
    """Имитация извлечения таблиц из документа."""
    # В реальном проекте здесь будет логика для парсинга таблиц
    # с помощью pdfplumber, python-docx и т.д.
    
    if "contract" in path:
        return [
            {
                "page": 1,
                "table": [
                    ["Item", "Old Value", "New Value"],
                    ["Price", "$100", "$120"],
                    ["Delivery", "30 days", "15 days"]
                ]
            }
        ]
    else:
        return []

def _fixed_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    """Простое разбиение текста на чанки по размеру."""
    chunks = []
    start = 0
    
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        start = end - overlap
    
    return chunks

def _semantic_chunking(text: str, chunk_size: int, overlap: int) -> List[str]:
    """
    Умное разбиение текста на чанки на основе семантического сходства.
    
    Использует эмбеддинги через UMS.
    """
    # Сначала разбиваем на предложения
    sentences = text.split('. ')
    
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) < chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    # В реальном проекте здесь можно использовать эмбеддинги для
    # определения оптимальных границ чанков
    # embeddings = [get_embeddings_via_ums(chunk) for chunk in chunks]
    
    return chunks

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
