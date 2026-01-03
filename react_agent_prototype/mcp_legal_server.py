"""
MCP Legal Server - FastAPI приложение для анализа юридических документов.

Функции:
- Сравнение двух документов (выявление различий)
- Анализ юридической значимости различий
- Генерация отчётов

Интегрирует UMS для использования:
- Embedding-модели (LaBSE) для сравнения текстов
- LLM-модели (Qwen-14B) для анализа значимости
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import json
from ums_client import get_embeddings_via_ums, generate_text_via_ums

app = FastAPI(title="MCP Legal Server", version="1.0.0")

# ============================================================================
# Request/Response Models
# ============================================================================

class CompareChunksRequest(BaseModel):
    old_text: str
    new_text: str
    threshold: float = 0.72

class DifferenceItem(BaseModel):
    type: str  # "ADDED", "DELETED", "MODIFIED"
    old_text: Optional[str] = None
    new_text: Optional[str] = None
    similarity_score: Optional[float] = None

class CompareChunksResponse(BaseModel):
    status: str
    differences: Optional[List[DifferenceItem]] = None
    error: Optional[str] = None

class AnalyzeImpactRequest(BaseModel):
    differences: List[DifferenceItem]

class ImpactAnalysis(BaseModel):
    difference: str
    is_critical: bool
    impact_description: str
    recommendation: str

class AnalyzeImpactResponse(BaseModel):
    status: str
    analysis: Optional[List[ImpactAnalysis]] = None
    error: Optional[str] = None

class GenerateReportRequest(BaseModel):
    analysis: List[ImpactAnalysis]
    format: str = "markdown"  # "markdown" или "json"

class GenerateReportResponse(BaseModel):
    status: str
    report: Optional[str] = None
    error: Optional[str] = None

# ============================================================================
# Endpoints
# ============================================================================

@app.get("/health")
async def health():
    """Health check endpoint."""
    return {"status": "healthy", "service": "mcp-legal-server"}

@app.post("/compare_chunks", response_model=CompareChunksResponse)
async def compare_chunks(request: CompareChunksRequest):
    """
    Сравнивает два текста и выявляет различия.
    
    Использует эмбеддинги через UMS для определения семантического сходства.
    """
    print(f"[LEGAL_SERVER] Comparing chunks (threshold={request.threshold})")
    
    try:
        # Получаем эмбеддинги через UMS
        old_embedding = get_embeddings_via_ums(request.old_text)
        new_embedding = get_embeddings_via_ums(request.new_text)
        
        # Вычисляем косинусное сходство
        similarity = _cosine_similarity(old_embedding, new_embedding)
        
        # Определяем тип различия
        if similarity < request.threshold:
            differences = [
                DifferenceItem(
                    type="MODIFIED",
                    old_text=request.old_text[:100],
                    new_text=request.new_text[:100],
                    similarity_score=similarity
                )
            ]
        else:
            differences = []
        
        # Также используем простой текстовый анализ для выявления конкретных изменений
        text_diffs = _extract_text_differences(request.old_text, request.new_text)
        differences.extend(text_diffs)
        
        return CompareChunksResponse(
            status="success",
            differences=differences
        )
    
    except Exception as e:
        return CompareChunksResponse(
            status="error",
            error=str(e)
        )

@app.post("/analyze_impact", response_model=AnalyzeImpactResponse)
async def analyze_impact(request: AnalyzeImpactRequest):
    """
    Анализирует юридическую значимость различий.
    
    Использует LLM через UMS для генерации анализа.
    """
    print(f"[LEGAL_SERVER] Analyzing impact of {len(request.differences)} differences")
    
    try:
        analysis = []
        
        for diff in request.differences:
            # Формируем промпт для LLM
            prompt = f"""
Проанализируй следующее изменение в юридическом документе и определи его значимость:

Тип изменения: {diff.type}
Старый текст: {diff.old_text}
Новый текст: {diff.new_text}

Ответь в формате JSON:
{{
    "is_critical": true/false,
    "impact": "описание влияния на договор",
    "recommendation": "рекомендация по действиям"
}}
"""
            
            # Используем LLM через UMS для анализа
            llm_response = generate_text_via_ums(prompt, max_tokens=256)
            
            # Парсим JSON-ответ (или используем значения по умолчанию)
            try:
                parsed = json.loads(llm_response)
                is_critical = parsed.get("is_critical", False)
                impact = parsed.get("impact", "Unknown impact")
                recommendation = parsed.get("recommendation", "Review required")
            except:
                # Если LLM вернул mock-ответ, используем эвристику
                is_critical = "Price" in str(diff.old_text) or "Delivery" in str(diff.old_text)
                impact = "Financial or operational change detected"
                recommendation = "Manual review recommended"
            
            analysis.append(ImpactAnalysis(
                difference=f"{diff.type}: {diff.old_text} -> {diff.new_text}",
                is_critical=is_critical,
                impact_description=impact,
                recommendation=recommendation
            ))
        
        return AnalyzeImpactResponse(
            status="success",
            analysis=analysis
        )
    
    except Exception as e:
        return AnalyzeImpactResponse(
            status="error",
            error=str(e)
        )

@app.post("/generate_report", response_model=GenerateReportResponse)
async def generate_report(request: GenerateReportRequest):
    """Генерирует отчёт на основе анализа различий."""
    print(f"[LEGAL_SERVER] Generating report (format={request.format})")
    
    try:
        if request.format == "markdown":
            report = _generate_markdown_report(request.analysis)
        else:
            report = json.dumps([a.dict() for a in request.analysis], indent=2)
        
        return GenerateReportResponse(
            status="success",
            report=report
        )
    
    except Exception as e:
        return GenerateReportResponse(
            status="error",
            error=str(e)
        )

# ============================================================================
# Helper Functions
# ============================================================================

def _cosine_similarity(vec1: list, vec2: list) -> float:
    """Вычисляет косинусное сходство между двумя векторами."""
    if len(vec1) != len(vec2):
        return 0.0
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a ** 2 for a in vec1) ** 0.5
    norm2 = sum(b ** 2 for b in vec2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)

def _extract_text_differences(old_text: str, new_text: str) -> List[DifferenceItem]:
    """Извлекает конкретные текстовые различия (простой анализ)."""
    differences = []
    
    # Простой анализ: ищем изменения в числах и ключевых словах
    old_words = set(old_text.split())
    new_words = set(new_text.split())
    
    added = new_words - old_words
    deleted = old_words - new_words
    
    for word in deleted:
        if word.replace("$", "").replace("days", "").isdigit() or word.isdigit():
            differences.append(DifferenceItem(
                type="DELETED",
                old_text=word,
                new_text=None
            ))
    
    for word in added:
        if word.replace("$", "").replace("days", "").isdigit() or word.isdigit():
            differences.append(DifferenceItem(
                type="ADDED",
                old_text=None,
                new_text=word
            ))
    
    return differences

def _generate_markdown_report(analysis: List[ImpactAnalysis]) -> str:
    """Генерирует отчёт в формате Markdown."""
    report = "# Анализ различий в юридическом документе\n\n"
    
    critical_count = sum(1 for a in analysis if a.is_critical)
    
    report += f"## Резюме\n"
    report += f"- **Всего различий:** {len(analysis)}\n"
    report += f"- **Критических:** {critical_count}\n\n"
    
    report += "## Детальный анализ\n\n"
    
    for i, item in enumerate(analysis, 1):
        critical_badge = "🔴 КРИТИЧНО" if item.is_critical else "🟡 ВНИМАНИЕ"
        report += f"### {i}. {critical_badge}\n\n"
        report += f"**Различие:** {item.difference}\n\n"
        report += f"**Влияние:** {item.impact_description}\n\n"
        report += f"**Рекомендация:** {item.recommendation}\n\n"
    
    return report

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8002)
