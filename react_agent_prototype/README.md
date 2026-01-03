# ReAct Agent Prototype with Unified Model Server

Этот прототип демонстрирует микросервисную архитектуру для агентной системы обработки документов с использованием **LangGraph**, **FastAPI** и **llama-cpp-python**.

## Архитектура

```
┌─────────────────────────────────────────────────────────────────┐
│                    ReAct Agent (LangGraph)                      │
│  - Orchestration Layer                                          │
│  - Thought → Action → Observation Loop                          │
└────────────┬──────────────────────────────────┬─────────────────┘
             │                                  │
             ▼                                  ▼
    ┌────────────────────┐          ┌────────────────────┐
    │ Document Server    │          │  Legal Server      │
    │ (FastAPI)          │          │  (FastAPI)         │
    │ Port: 8001         │          │  Port: 8002        │
    │                    │          │                    │
    │ - load_document    │          │ - compare_chunks   │
    │ - extract_tables   │          │ - analyze_impact   │
    │ - smart_chunk      │          │ - generate_report  │
    └────────┬───────────┘          └────────┬───────────┘
             │                               │
             └───────────────┬───────────────┘
                             ▼
                    ┌────────────────────┐
                    │ UMS Client         │
                    │ (HTTP)             │
                    │                    │
                    │ Requests to UMS    │
                    └────────┬───────────┘
                             │
                             ▼
                    ┌────────────────────┐
                    │ Unified Model      │
                    │ Server (UMS)       │
                    │ Port: 8090         │
                    │                    │
                    │ - Dynamic Loading  │
                    │ - Smart Swapping   │
                    │ - CPU/GPU/Hybrid   │
                    └────────┬───────────┘
                             │
                             ▼
                    ┌────────────────────┐
                    │ llama-cpp-python   │
                    │ (llama-server)     │
                    │                    │
                    │ Models:            │
                    │ - Qwen-14B-LLM     │
                    │ - Qwen-VL-Vision   │
                    │ - LaBSE-Embedding  │
                    └────────────────────┘
```

## Компоненты

### 1. **ReAct Agent** (`react_agent_http.py`)
- Главный оркестратор, использующий LangGraph
- Реализует цикл Thought → Action → Observation
- Делает HTTP-запросы к MCP-серверам
- Управляет логикой выполнения задач

### 2. **MCP Document Server** (`mcp_document_server.py`)
- FastAPI приложение на порту 8001
- Функции:
  - `/load_document` - загрузка документов
  - `/extract_tables` - извлечение таблиц
  - `/smart_chunk` - умное разбиение текста на чанки
- Интегрирует UMS-клиент для OCR и эмбеддингов

### 3. **MCP Legal Server** (`mcp_legal_server.py`)
- FastAPI приложение на порту 8002
- Функции:
  - `/compare_chunks` - сравнение двух текстов
  - `/analyze_impact` - анализ юридической значимости различий
  - `/generate_report` - генерация отчётов
- Интегрирует UMS-клиент для эмбеддингов и LLM-анализа

### 4. **UMS Client** (`ums_client.py`)
- HTTP-клиент для общения с Unified Model Server
- Предоставляет функции:
  - `generate_text_via_ums()` - генерация текста через LLM
  - `get_embeddings_via_ums()` - получение эмбеддингов
  - `process_vision_via_ums()` - обработка изображений через Vision-модель

### 5. **Unified Model Server** (`unified_model_server.py`)
- Управляет загрузкой/выгрузкой моделей
- Поддерживает режимы: GPU, CPU, Hybrid
- Реализует Smart Swapping для оптимизации VRAM
- В текущем прототипе это заглушка; в реальном проекте будет отдельным сервисом

## Установка

```bash
# 1. Перейти в директорию проекта
cd react_agent_prototype

# 2. Создать виртуальное окружение
python3.11 -m venv venv
source venv/bin/activate

# 3. Установить зависимости
pip install -r requirements.txt
```

## Запуск

### Вариант 1: Автоматический запуск всех компонентов (с tmux)

```bash
./run_all.sh
```

Это запустит все компоненты в отдельных окнах tmux:
- `doc-server` - Document Server
- `legal-server` - Legal Server
- `react-agent` - ReAct Agent
- `monitor` - Окно для мониторинга

### Вариант 2: Запуск с тестом системы

```bash
python3.11 test_system.py
```

Этот скрипт:
1. Запускает MCP-серверы в фоне
2. Ожидает их инициализации
3. Запускает ReAct-агент
4. Собирает результаты
5. Останавливает все процессы

### Вариант 3: Ручной запуск компонентов

**Терминал 1 - Document Server:**
```bash
source venv/bin/activate
python3.11 mcp_document_server.py
```

**Терминал 2 - Legal Server:**
```bash
source venv/bin/activate
python3.11 mcp_legal_server.py
```

**Терминал 3 - ReAct Agent:**
```bash
source venv/bin/activate
python3.11 react_agent_http.py
```

## Тестирование

### Проверка доступности серверов

```bash
# Document Server
curl http://localhost:8001/health

# Legal Server
curl http://localhost:8002/health
```

### Тестирование Document Server

```bash
curl -X POST http://localhost:8001/load_document \
  -H "Content-Type: application/json" \
  -d '{"path": "contract_old.pdf"}'
```

### Тестирование Legal Server

```bash
curl -X POST http://localhost:8002/compare_chunks \
  -H "Content-Type: application/json" \
  -d '{
    "old_text": "Document A: Contract text (old version). Clause 1: Price is $100. Clause 2: Delivery in 30 days.",
    "new_text": "Document B: Contract text (new version). Clause 1: Price is $120. Clause 2: Delivery in 15 days."
  }'
```

## Примеры использования

### Пример 1: Сравнение двух документов

```python
from react_agent_http import create_react_graph

app = create_react_graph()

initial_state = {
    "user_query": "Сравни contract_old.pdf и contract_new.pdf",
    "chat_history": [],
    "available_tools": ["document_server.load_document", "legal_server.compare_chunks"],
    "current_thought": "",
    "planned_action": {},
    "observation": "",
    "final_answer": "",
    "is_finished": False
}

for step in app.stream(initial_state):
    print(step)
```

## Структура проекта

```
react_agent_prototype/
├── react_agent.py                 # Оригинальный ReAct-агент (заглушки)
├── react_agent_http.py            # ReAct-агент с HTTP-клиентами
├── mcp_document_server.py         # Document Server (FastAPI)
├── mcp_legal_server.py            # Legal Server (FastAPI)
├── mcp_servers_mock.py            # Mock-серверы (для справки)
├── ums_client.py                  # UMS HTTP-клиент
├── unified_model_server.py        # UMS (заглушка)
├── models_config.py               # Конфигурация моделей
├── test_system.py                 # Скрипт для тестирования системы
├── run_all.sh                     # Скрипт для запуска всех компонентов
├── requirements.txt               # Зависимости Python
├── .gitignore                     # Git ignore
└── README.md                      # Этот файл
```

## Следующие шаги

1. **Реализация реального UMS** - создать отдельный сервис с управлением моделями
2. **Интеграция llama-cpp-python** - подключить реальные модели
3. **Добавление новых MCP-серверов** - Procurement Server, Analytics Server и т.д.
4. **Оптимизация производительности** - кэширование, батчинг запросов
5. **Развертывание в Docker** - контейнеризация всех компонентов

## Лицензия

MIT

## Автор

Разработано как прототип микросервисной архитектуры для агентной системы обработки документов.
