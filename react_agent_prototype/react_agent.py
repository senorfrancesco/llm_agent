import json
from typing import TypedDict, List, Dict, Any, Union
from langgraph.graph import StateGraph, END
from unified_model_server import generate_text

# ============================================================================
# 1. ReAct Agent State (пользовательский план)
# ============================================================================

class ReActState(TypedDict):
    """
    Состояние для ReAct-агента.
    Соответствует структуре, предложенной в плане рефакторинга.
    """
    user_query: str              # Исходный запрос пользователя
    chat_history: List[str]      # История: [thought, action, observation]
    available_tools: List[str]   # Список доступных MCP инструментов (для промпта)
    current_thought: str         # Текущее рассуждение LLM
    planned_action: Dict[str, Any] # Следующее действие {"tool": "...", "params": {...}}
    observation: Any             # Результат выполнения последнего action
    final_answer: str            # Финальный ответ пользователю
    is_finished: bool            # Флаг завершения

# ============================================================================
# 2. Tool Definitions (Интерфейсы MCP-серверов)
# ============================================================================

# В реальном проекте эти функции будут делать HTTP-запросы к MCP-серверам.
# Здесь они имитируют эту логику.

def document_server_load_document(path: str) -> str:
    """
    MCP Document Server: Загружает документ, возвращает его текстовое представление.
    Имитирует вызов: POST http://document-server:8000/load_document
    """
    print(f"\n[TOOL] Document Server: Loading document from {path}...")
    # Имитация вызова UMS для OCR/парсинга
    # text_data = process_vision(path, "Extract all text from the document.")
    
    # Mock-ответ
    if "contract_old" in path:
        return "Document A: Contract text (old version). Clause 1: Price is $100. Clause 2: Delivery in 30 days."
    elif "contract_new" in path:
        return "Document B: Contract text (new version). Clause 1: Price is $120. Clause 2: Delivery in 15 days."
    else:
        return f"Document {path} loaded successfully."

def legal_server_compare_chunks(old_text: str, new_text: str) -> str:
    """
    MCP Legal Server: Сравнивает два текста, возвращает различия.
    Имитирует вызов: POST http://legal-server:8000/compare_chunks
    """
    print("\n[TOOL] Legal Server: Comparing documents...")
    # Имитация вызова UMS для эмбеддингов и сравнения
    # embeddings_old = get_embeddings(old_text)
    # embeddings_new = get_embeddings(new_text)
    
    # Mock-ответ
    diff = []
    if "100" in old_text and "120" in new_text:
        diff.append("Price changed from $100 to $120.")
    if "30 days" in old_text and "15 days" in new_text:
        diff.append("Delivery time reduced from 30 to 15 days.")
        
    return json.dumps({"differences": diff})

def legal_server_analyze_impact(differences: str) -> str:
    """
    MCP Legal Server: Анализирует юридическую значимость различий.
    Имитирует вызов: POST http://legal-server:8000/analyze_impact
    """
    print("\n[TOOL] Legal Server: Analyzing impact...")
    # Имитация вызова UMS для LLM-анализа
    # analysis_prompt = f"Analyze the legal impact of these differences: {differences}"
    # analysis_result = generate_text(analysis_prompt)
    
    # Mock-ответ
    if "Price changed" in differences:
        return "The price change is a critical financial impact. The delivery time reduction is a positive operational change."
    else:
        return "No critical impact found."

# Словарь доступных инструментов
TOOLS = {
    "document_server.load_document": document_server_load_document,
    "legal_server.compare_chunks": legal_server_compare_chunks,
    "legal_server.analyze_impact": legal_server_analyze_impact,
}

# ============================================================================
# 3. LLM Call (Decision Node)
# ============================================================================

def get_tool_schema() -> str:
    """Генерирует схему инструментов для промпта LLM."""
    schema = "TOOLS:\n"
    for name, func in TOOLS.items():
        schema += f"1. {name}(path: str) -> str\n" # Упрощенная схема
    return schema

def get_react_prompt(state: ReActState) -> str:
    """Формирует промпт для LLM, включая историю ReAct."""
    
    # Формат истории: Thought -> Action -> Observation
    history_str = "\n".join(state["chat_history"])
    
    prompt = f"""
Ты — AI-агент для анализа документов. Твоя задача — выполнить запрос пользователя, используя доступные инструменты.
Ты должен строго следовать формату ответа ReAct: сначала рассуждение (Thought), затем действие (Action) или финальный ответ (Final Answer).

{get_tool_schema()}

ФОРМАТ ОТВЕТА:
Thought: <твоё рассуждение о следующем шаге>
Action: <tool_name(param1="value", param2="value")> ИЛИ Final Answer: <ответ пользователю>

ИСТОРИЯ ВЫПОЛНЕНИЯ:
{history_str}

ЗАПРОС ПОЛЬЗОВАТЕЛЯ:
{state["user_query"]}

ТВОЙ СЛЕДУЮЩИЙ ШАГ:
"""
    return prompt.strip()

def call_llm(state: ReActState) -> ReActState:
    """
    Узел LangGraph: LLM принимает решение (Thought + Action/Final Answer).
    """
    print("\n[LLM] Calling LLM for decision...")
    
    prompt = get_react_prompt(state)
    
    # Имитация вызова UMS для LLM
    # response = generate_text(prompt)
    
    # Mock-ответ LLM для демонстрации логики
    if not state["chat_history"]:
        # Первый шаг: загрузить первый документ
        response = """Thought: Пользователь хочет сравнить два документа. Сначала мне нужно загрузить первый документ с помощью document_server.load_document.
Action: document_server.load_document(path="contract_old.pdf")"""
    elif "Document A" in state["observation"] and "Document B" not in state["observation"]:
        # Второй шаг: загрузить второй документ
        response = f"""Thought: Первый документ загружен. Теперь нужно загрузить второй документ для сравнения.
Action: document_server.load_document(path="contract_new.pdf")"""
    elif "Document B" in state["observation"] and "differences" not in state["observation"]:
        # Третий шаг: сравнить документы
        response = f"""Thought: Оба документа загружены. Теперь я должен сравнить их, используя legal_server.compare_chunks.
Action: legal_server.compare_chunks(old_text="{state["chat_history"][-3]}", new_text="{state["observation"]}")"""
    elif "differences" in state["observation"] and "critical financial impact" not in state["observation"]:
        # Четвертый шаг: проанализировать различия
        response = f"""Thought: Различия найдены. Теперь нужно проанализировать их юридическую значимость с помощью legal_server.analyze_impact.
Action: legal_server.analyze_impact(differences='{state["observation"]}')"""
    else:
        # Финальный шаг: ответить
        response = f"""Thought: Анализ завершен. Я могу предоставить финальный ответ, используя результаты анализа: {state["observation"]}.
Final Answer: В результате сравнения обнаружены критические изменения: {state["observation"]}"""

    # Парсинг ответа LLM
    thought = ""
    action = {}
    final_answer = ""
    
    if "Thought:" in response:
        thought = response.split("Thought:")[1].split("Action:")[0].split("Final Answer:")[0].strip()
    
    if "Action:" in response:
        action_str = response.split("Action:")[1].strip()
        # Простой парсинг Action: tool_name(param1="value", ...)
        tool_name = action_str.split("(")[0]
        params_str = action_str.split("(")[1].split(")")[0]
        
        params = {}
        # Очень простой парсинг параметров (только для демонстрации)
        for part in params_str.split(','):
            if '=' in part:
                key, value = part.split('=', 1)
                params[key.strip()] = value.strip().strip('"').strip("'")
        
        action = {"tool": tool_name, "params": params}
    
    if "Final Answer:" in response:
        final_answer = response.split("Final Answer:")[1].strip()
    
    # Обновление состояния
    new_history = state["chat_history"] + [f"Thought: {thought}"]
    
    if action:
        new_history.append(f"Action: {action['tool']}({json.dumps(action['params'])})")
    
    return {
        "chat_history": new_history,
        "current_thought": thought,
        "planned_action": action,
        "final_answer": final_answer,
        "is_finished": bool(final_answer)
    }

# ============================================================================
# 4. Tool Call Node
# ============================================================================

def call_tool(state: ReActState) -> ReActState:
    """
    Узел LangGraph: Вызывает выбранный инструмент.
    """
    action = state["planned_action"]
    tool_name = action["tool"]
    params = action["params"]
    
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    tool_func = TOOLS[tool_name]
    
    try:
        # Вызов инструмента с параметрами
        observation = tool_func(**params)
        
        # Обновление истории
        new_history = state["chat_history"] + [f"Observation: {observation}"]
        
        return {
            "chat_history": new_history,
            "observation": observation,
            "planned_action": {} # Сброс действия
        }
    except Exception as e:
        error_msg = f"Tool {tool_name} failed with error: {e}"
        print(f"[TOOL ERROR] {error_msg}")
        
        # Обновление истории с ошибкой
        new_history = state["chat_history"] + [f"Observation: Tool failed: {error_msg}"]
        
        # Возвращаемся к LLM, чтобы он решил, что делать дальше
        return {
            "chat_history": new_history,
            "observation": error_msg,
            "planned_action": {}
        }

# ============================================================================
# 5. Conditional Edge
# ============================================================================

def should_continue(state: ReActState) -> str:
    """
    Определяет, должен ли граф продолжаться (вызов инструмента) или завершиться.
    """
    if state["is_finished"]:
        return "end"
    
    if state["planned_action"]:
        return "continue"
    
    # Если нет Final Answer и нет Planned Action, то это ошибка или нужно
    # снова вызвать LLM (например, после ошибки инструмента)
    return "llm"

# ============================================================================
# 6. Graph Definition
# ============================================================================

def create_react_graph():
    """Создает и компилирует граф ReAct-агента."""
    
    workflow = StateGraph(ReActState)
    
    # 1. Добавляем узлы
    workflow.add_node("llm", call_llm)
    workflow.add_node("tool", call_tool)
    
    # 2. Устанавливаем начальную точку
    workflow.set_entry_point("llm")
    
    # 3. Определяем условные переходы
    workflow.add_conditional_edges(
        "llm",          # Откуда
        should_continue, # Функция-маршрутизатор
        {               # Куда
            "continue": "tool", # Если есть Action, идем к инструменту
            "end": END,         # Если есть Final Answer, завершаем
            "llm": "llm"        # Если ошибка, снова к LLM (для обработки)
        }
    )
    
    # 4. Определяем обычный переход (после инструмента всегда идем к LLM)
    workflow.add_edge("tool", "llm")
    
    return workflow.compile()

# ============================================================================
# 7. Execution Example
# ============================================================================

if __name__ == "__main__":
    # Имитация запуска llama-server для LLM (в реальном проекте это будет
    # происходить внутри UMS, но для демонстрации LangGraph мы мокаем)
    
    app = create_react_graph()
    
    # Исходное состояние
    initial_state = {
        "user_query": "Сравни два документа: contract_old.pdf и contract_new.pdf, и оцени юридическую значимость различий.",
        "chat_history": [],
        "available_tools": list(TOOLS.keys()),
        "current_thought": "",
        "planned_action": {},
        "observation": "",
        "final_answer": "",
        "is_finished": False
    }
    
    print("--- Starting ReAct Agent Execution ---")
    
    # Запуск графа
    for step in app.stream(initial_state):
        print("-" * 50)
        # Выводим только ключевые изменения
        for key, value in step.items():
            if key == "llm":
                print(f"NODE: {key}")
                print(f"  Thought: {value.get('current_thought')}")
                if value.get('planned_action'):
                    print(f"  Action: {value['planned_action']['tool']}({json.dumps(value['planned_action']['params'])})")
            elif key == "tool":
                print(f"NODE: {key}")
                print(f"  Observation: {value.get('observation')[:100]}...")
            elif key == '__end__':
                print(f"NODE: {key}")
                print(f"FINAL ANSWER: {step[key]['final_answer']}")
    
    print("--- ReAct Agent Execution Finished ---")
