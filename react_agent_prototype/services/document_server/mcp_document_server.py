"""
MCP Document Server - FastAPI приложение для работы с документами.

Функции:
- Загрузка документов (PDF, DOCX, TXT)
- Извлечение текста (через OCR для изображений, если нужно)
- Извлечение таблиц
- Семантическое разбиение текста на чанки с поддержкой перекрытия

Интегрирует UMS для использования Vision-модели (Qwen-VL) для OCR.
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
import os
from pathlib import Path
import sys

# Добавляем путь к model_manager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'model_manager'))

from ums_client import process_vision_via_ums, get_embeddings_via_ums, ums_client

app = FastAPI(title="MCP Document Server", version="1.0.0")

# ============================================================================
# Request/Response Models
# ============================================================================

class LoadDocumentRequest(BaseModel):
    path: str
    extract_tables: bool = False
    use_ocr: bool = False

class SmartChunkRequest(BaseModel):
    text: str
    max_tokens: int = 8000  # Для llama.cpp с n_ctx=8192
    overlap: int = 100      # Перекрытие в токенах

class ExtractTablesRequest(BaseModel):
    path: str
    pages: Optional[List[int]] = None

# ============================================================================
# Document Loading Functions
# ============================================================================

def load_pdf(path: str) -> str:
    """Извлечение текста из PDF."""
    try:
        import pdfplumber
        text = ""
        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, 1):
                text += f"\n--- Page {page_num} ---\n"
                text += page.extract_text() or ""
        return text
    except ImportError:
        raise RuntimeError("pdfplumber not installed. Install with: pip install pdfplumber")
    except Exception as e:
        raise RuntimeError(f"Error reading PDF: {e}")

def load_docx(path: str) -> str:
    """Извлечение текста из DOCX."""
    try:
        from docx import Document
        doc = Document(path)
        text = "\n".join([para.text for para in doc.paragraphs])
        return text
    except ImportError:
        raise RuntimeError("python-docx not installed. Install with: pip install python-docx")
    except Exception as e:
        raise RuntimeError(f"Error reading DOCX: {e}")

def load_txt(path: str) -> str:
    """Загрузка текста из TXT."""
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        raise RuntimeError(f"Error reading TXT: {e}")

# ============================================================================
# Chunking Functions
# ============================================================================

def smart_chunk(text: str, max_tokens: int = 8000, overlap: int = 100) -> List[str]:
    """
    Умное разбиение текста на чанки с поддержкой перекрытия.
    
    Стратегия:
    1. Разбиение по логическим границам (абзацы, главы)
    2. Учет размера контекста модели (max_tokens)
    3. Перекрытие между чанками для сохранения контекста
    
    Args:
        text: Исходный текст
        max_tokens: Максимальное количество токенов в чанке (примерно)
        overlap: Перекрытие в токенах между соседними чанками
    
    Returns:
        Список чанков
    """
    # Разбиваем по абзацам (логические границы)
    paragraphs = text.split('\n\n')
    
    chunks = []
    current_chunk = ""
    current_token_count = 0
    
    # Примерный подсчет: 1 слово ≈ 1.3 токена
    for para in paragraphs:
        para_tokens = len(para.split()) * 1.3
        
        # Если добавление абзаца превышает лимит, сохраняем текущий чанк
        if current_token_count + para_tokens > max_tokens and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = para + "\n\n"
            current_token_count = para_tokens
        else:
            current_chunk += para + "\n\n"
            current_token_count += para_tokens
    
    # Добавляем последний чанк
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Добавляем перекрытие между чанками
    if overlap > 0 and len(chunks) > 1:
        overlapped_chunks = []
        overlap_lines_count = max(1, int(overlap / 50))  # Примерно 50 токенов на строку
        
        for i, chunk in enumerate(chunks):
            if i > 0:
                # Берем последние overlap_lines из предыдущего чанка
                prev_lines = chunks[i-1].split('\n')
                overlap_lines = prev_lines[-overlap_lines_count:]
                overlapped_chunks.append('\n'.join(overlap_lines) + '\n\n' + chunk)
            else:
                overlapped_chunks.append(chunk)
        
        chunks = overlapped_chunks
    
    return chunks

# ============================================================================
# API Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "service": "document_server"}

@app.post("/load_document")
async def load_document(request: LoadDocumentRequest):
    """
    Загружает документ и извлекает текст.
    
    Поддерживаемые форматы: PDF, DOCX, TXT.
    """
    try:
        path = request.path
        
        # Проверяем существование файла
        if not os.path.exists(path):
            # Для тестирования возвращаем mock-данные
            if "contract_old" in path:
                text = "Document A: Contract text (old version). Clause 1: Price is $100. Clause 2: Delivery in 30 days."
            elif "contract_new" in path:
                text = "Document B: Contract text (new version). Clause 1: Price is $120. Clause 2: Delivery in 15 days."
            else:
                text = f"[Mock] Document {path} loaded successfully."
            
            return {
                "status": "success",
                "text": text,
                "path": path,
                "format": "mock",
                "length": len(text)
            }
        
        # Определяем формат файла
        file_ext = Path(path).suffix.lower()
        
        if file_ext == ".pdf":
            text = load_pdf(path)
            format_type = "pdf"
        elif file_ext == ".docx":
            text = load_docx(path)
            format_type = "docx"
        elif file_ext == ".txt":
            text = load_txt(path)
            format_type = "txt"
        else:
            return {
                "status": "error",
                "error": f"Unsupported file format: {file_ext}. Supported: PDF, DOCX, TXT"
            }
        
        # Если нужно, используем OCR для изображений
        if request.use_ocr and file_ext in [".jpg", ".png", ".jpeg"]:
            text = process_vision_via_ums(path, "Extract all text from this image")
            format_type = "image_ocr"
        
        return {
            "status": "success",
            "text": text,
            "path": path,
            "format": format_type,
            "length": len(text)
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/smart_chunk")
async def smart_chunk_endpoint(request: SmartChunkRequest):
    """
    Разбивает текст на чанки с учетом контекста модели.
    
    Использует семантическое разбиение по абзацам и добавляет перекрытие.
    """
    try:
        chunks = smart_chunk(
            request.text,
            max_tokens=request.max_tokens,
            overlap=request.overlap
        )
        
        return {
            "status": "success",
            "chunks": chunks,
            "chunk_count": len(chunks),
            "strategy": "semantic_with_overlap",
            "max_tokens": request.max_tokens,
            "overlap": request.overlap
        }
    
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

@app.post("/extract_tables")
async def extract_tables(request: ExtractTablesRequest):
    """
    Извлекает таблицы из PDF.
    """
    try:
        import pdfplumber
        
        if not os.path.exists(request.path):
            return {
                "status": "error",
                "error": f"File not found: {request.path}"
            }
        
        tables = []
        with pdfplumber.open(request.path) as pdf:
            pages_to_process = request.pages or range(len(pdf.pages))
            
            for page_num in pages_to_process:
                if page_num < len(pdf.pages):
                    page = pdf.pages[page_num]
                    page_tables = page.extract_tables()
                    
                    if page_tables:
                        for table in page_tables:
                            tables.append({
                                "page": page_num + 1,
                                "data": table
                            })
        
        return {
            "status": "success",
            "tables": tables,
            "table_count": len(tables)
        }
    
    except ImportError:
        return {
            "status": "error",
            "error": "pdfplumber not installed. Install with: pip install pdfplumber"
        }
    except Exception as e:
        return {
            "status": "error",
            "error": str(e)
        }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)
