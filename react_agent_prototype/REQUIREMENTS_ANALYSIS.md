# Анализ требований из 'ЗадачаподLLM.txt' и текущей реализации

## Обзор требований

Документ описывает две основные задачи:

1.  **Сравнение нормативных документов** (50+ страниц): Необходимо не просто выявить различия, но и предоставить комментарии модели о том, как изменился смысл текста, включая влияние мелких изменений (например, запятая может изменить смысл предложения).

2.  **Анализ коммерческих предложений**: Анализ файлов (PDF, DOCX, фото) с предложениями товаров и оценка их соответствия заданным параметрам.

### Ключевые ограничения

*   **Оборудование:** RTX 3060 (12GB VRAM) + 32GB RAM + Ryzen 7 5800H
*   **Сеть:** Изолированная (без интернета)
*   **Форматы:** PDF (приоритет), DOCX, фото
*   **Адаптивная оптимизация:** Система должна адаптироваться к доступным ресурсам

### Технические вызовы

1.  **Ограничение контекста:** Большие документы (50+ страниц) превышают размер контекста моделей:
    *   `llama.cpp`: n_ctx=8192 (гарантированно), n_ctx=16384 (может работать)
    *   `vLLM`: полные 32K токенов с поддержкой YARN для расширения контекста
    *   Требуется **чанкинг** (разбиение документа на части)

2.  **Выбор библиотеки:** Разные библиотеки имеют разные возможности:
    *   `llama.cpp`: Хороша для CPU/GPU, но ограничен контекстом
    *   `vLLM`: Требует GPU, но поддерживает полный контекст
    *   `transformers`: Требует больше памяти, но максимальная гибкость

## Текущая реализация

### Что реализовано ✅

| Требование | Статус | Реализация |
| :--- | :--- | :--- |
| **Микросервисная архитектура** | ✅ | ReAct Agent + MCP Servers (Document, Legal) |
| **Управление моделями (UMS)** | ✅ | Unified Model Server с динамической загрузкой/выгрузкой |
| **Поддержка режимов GPU/CPU/Hybrid** | ✅ | UMS поддерживает `--n_gpu_layers` для адаптации |
| **HTTP API для инструментов** | ✅ | MCP-серверы как FastAPI приложения с HTTP API |
| **Инструменты для агента** | ✅ | Document Server и Legal Server как инструменты |
| **Локальные модели (llama-cpp-python)** | ✅ | UMS запускает `llama-server` локально |

### Что требует доработки ⚠️

| Требование | Статус | Необходимые действия |
| :--- | :--- | :--- |
| **Чанкинг больших документов** | ⚠️ | Реализовать `smart_chunk` в Document Server с поддержкой семантического анализа |
| **Поддержка PDF** | ⚠️ | Добавить `pdfplumber` или `PyPDF2` для извлечения текста из PDF |
| **Поддержка DOCX** | ❌ | Добавить `python-docx` для работы с DOCX |
| **Поддержка фото/OCR** | ⚠️ | Использовать Vision-модель (Qwen-VL) через UMS |
| **Детальный анализ изменений** | ⚠️ | Улучшить `/analyze_impact` в Legal Server для выявления семантических изменений |
| **Переход на vLLM** | ❌ | Модификация UMS для поддержки vLLM (для больших контекстов) |
| **RAG система** | ❌ | Добавить RAG для лучшего контекста при анализе документов |

## Рекомендации по доработке

### 1. Реализация `smart_chunk` в Document Server

Текущая реализация использует простое разбиение. Необходимо:

```python
def smart_chunk(text: str, max_tokens: int = 8000, overlap: int = 500):
    """
    Умное разбиение документа на чанки.
    
    Стратегия:
    1. Разбиение по логическим границам (абзацы, главы)
    2. Семантический анализ для определения связанности
    3. Перекрытие чанков для сохранения контекста
    """
    # Разбиение по абзацам
    paragraphs = text.split('\n\n')
    chunks = []
    current_chunk = ""
    
    for para in paragraphs:
        tokens = len(para.split())  # Примерный подсчет
        if len(current_chunk.split()) + tokens < max_tokens:
            current_chunk += para + "\n\n"
        else:
            if current_chunk:
                chunks.append(current_chunk)
            current_chunk = para + "\n\n"
    
    if current_chunk:
        chunks.append(current_chunk)
    
    # Добавляем перекрытие
    overlapped_chunks = []
    for i, chunk in enumerate(chunks):
        if i > 0:
            prev_lines = chunks[i-1].split('\n')[-overlap//100:]
            overlapped_chunks.append('\n'.join(prev_lines) + '\n' + chunk)
        else:
            overlapped_chunks.append(chunk)
    
    return overlapped_chunks
```

### 2. Добавление поддержки PDF и DOCX

```python
# В Document Server
from pdfplumber import PDF
from docx import Document

def load_pdf(path: str) -> str:
    """Извлечение текста из PDF."""
    text = ""
    with PDF.open(path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() + "\n"
    return text

def load_docx(path: str) -> str:
    """Извлечение текста из DOCX."""
    doc = Document(path)
    return "\n".join([para.text for para in doc.paragraphs])
```

### 3. Улучшение анализа изменений в Legal Server

Текущая реализация использует простое сравнение эмбеддингов. Необходимо:

```python
def analyze_semantic_changes(old_text: str, new_text: str):
    """
    Анализирует семантические изменения между текстами.
    
    Стратегия:
    1. Выделение ключевых фраз (NER, TF-IDF)
    2. Сравнение эмбеддингов фраз
    3. Анализ синтаксических изменений
    4. Оценка влияния на смысл
    """
    # Использовать LLM для анализа
    prompt = f"""
    Сравни два текста и определи, как изменился смысл:
    
    СТАРЫЙ ТЕКСТ:
    {old_text}
    
    НОВЫЙ ТЕКСТ:
    {new_text}
    
    Для каждого изменения укажи:
    1. Что изменилось (строка/абзац)
    2. Как изменился смысл
    3. Критичность изменения (CRITICAL/MODERATE/MINOR)
    4. Влияние на интерпретацию документа
    """
    
    # Отправить в LLM через UMS
    response = ums_client.infer("qwen-14b-llm", {"prompt": prompt})
    return response
```

### 4. Переход на vLLM (для больших контекстов)

Когда потребуется поддержка 32K+ токенов:

```python
# В unified_model_server.py
class VLLMBackend:
    """Backend для vLLM (поддержка больших контекстов)."""
    
    def __init__(self, model_path: str):
        from vllm import LLM
        self.llm = LLM(model=model_path, tensor_parallel_size=1, gpu_memory_utilization=0.9)
    
    def infer(self, prompt: str, max_tokens: int = 8192):
        from vllm import SamplingParams
        sampling_params = SamplingParams(temperature=0.7, max_tokens=max_tokens)
        outputs = self.llm.generate(prompt, sampling_params)
        return outputs[0].outputs[0].text
```

## План реализации

### Фаза 1: Базовая поддержка документов
1.  Реализовать `smart_chunk` в Document Server
2.  Добавить поддержку PDF и DOCX
3.  Протестировать на 50-страничных документах

### Фаза 2: Улучшение анализа
1.  Улучшить `/analyze_impact` для выявления семантических изменений
2.  Добавить детальный комментарий модели о каждом изменении
3.  Реализовать выделение критических изменений

### Фаза 3: Оптимизация и масштабирование
1.  Добавить RAG систему для лучшего контекста
2.  Реализовать кэширование эмбеддингов
3.  Переход на vLLM для поддержки больших контекстов

### Фаза 4: Расширение функциональности
1.  Добавить Procurement Server для анализа коммерческих предложений
2.  Реализовать Vision-анализ для фотографий товаров
3.  Добавить самообучение (fine-tuning) на основе пользовательских оценок
