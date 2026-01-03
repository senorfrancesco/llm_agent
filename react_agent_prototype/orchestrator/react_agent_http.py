"""
ReAct Agent with HTTP Clients for MCP Servers.

Этот агент использует реальные HTTP-запросы к MCP-серверам вместо функций-заглушек.
Это позволяет тестировать полную систему на одной машине.
"""

import json
import requests
from typing import TypedDict, List, Dict, Any
import sys
import os

# Добавляем путь к model_manager
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'services', 'model_manager'))

from langgraph.graph import StateGraph, END

# ============================================================================
# Configuration
# ============================================================================

MCP_DOCUMENT_SERVER_URL = "http://localhost:8001"
MCP_LEGAL_SERVER_URL = "http://localhost:8002"

# ============================================================================
# ReAct State
# ============================================================================

class ReActState(TypedDict):
    """Состояние для ReAct-агента."""
    user_query: str
    chat_history: List[str]
    available_tools: List[str]
    current_thought: str
    planned_action: Dict[str, Any]
    observation: Any
    final_answer: str
    is_finished: bool

# ============================================================================
# Tool Definitions (HTTP Clients)
# ============================================================================

def document_server_load_document(path: str) -> str:
    """Загружает документ через HTTP-запрос к Document Server."""
    print(f"\n[TOOL] Document Server: Loading document from {path}...")
    
    try:
        response = requests.post(
            f"{MCP_DOCUMENT_SERVER_URL}/load_document",
            json={"path": path, "extract_tables": False, "use_ocr": False},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "success":
            return data.get("text", "Document loaded but no text returned.")
        else:
            return f"Error loading document: {data.get('error', 'Unknown error')}"
    
    except requests.exceptions.RequestException as e:
        print(f"[TOOL ERROR] Failed to connect to Document Server: {e}")
        # Fallback на mock-ответ для демонстрации
        if "contract_old" in path:
            return "Document A: Contract text (old version). Clause 1: Price is $100. Clause 2: Delivery in 30 days."
        elif "contract_new" in path:
            return "Document B: Contract text (new version). Clause 1: Price is $120. Clause 2: Delivery in 15 days."
        else:
            return f"[MOCK] Document {path} loaded successfully."

def legal_server_compare_chunks(old_text: str, new_text: str) -> str:
    """Сравнивает два текста через HTTP-запрос к Legal Server."""
    print("\n[TOOL] Legal Server: Comparing documents...")
    
    try:
        response = requests.post(
            f"{MCP_LEGAL_SERVER_URL}/compare_chunks",
            json={"old_text": old_text, "new_text": new_text, "threshold": 0.72},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "success":
            differences = data.get("differences", [])
            return json.dumps({"differences": differences})
        else:
            return json.dumps({"error": data.get("error", "Unknown error")})
    
    except requests.exceptions.RequestException as e:
        print(f"[TOOL ERROR] Failed to connect to Legal Server: {e}")
        # Fallback на mock-ответ
        diff = []
        if "100" in old_text and "120" in new_text:
            diff.append({"type": "MODIFIED", "detail": "Price changed from $100 to $120."})
        if "30 days" in old_text and "15 days" in new_text:
            diff.append({"type": "MODIFIED", "detail": "Delivery time reduced from 30 to 15 days."})
        return json.dumps({"differences": diff})

def legal_server_analyze_impact(differences: str) -> str:
    """Анализирует юридическую значимость различий через Legal Server."""
    print("\n[TOOL] Legal Server: Analyzing impact...")
    
    try:
        # Парсим differences, если это JSON-строка
        if isinstance(differences, str):
            try:
                diff_data = json.loads(differences)
                differences_list = diff_data.get("differences", [])
            except:
                differences_list = []
        else:
            differences_list = differences if isinstance(differences, list) else []
        
        # Преобразуем в объекты DifferenceItem
        diff_items = []
        for diff in differences_list:
            if isinstance(diff, dict):
                diff_items.append({
                    "type": diff.get("type", "MODIFIED"),
                    "old_text": diff.get("old_text", ""),
                    "new_text": diff.get("new_text", ""),
                    "similarity_score": diff.get("similarity_score")
                })
        
        response = requests.post(
            f"{MCP_LEGAL_SERVER_URL}/analyze_impact",
            json={"differences": diff_items},
            timeout=10
        )
        response.raise_for_status()
        data = response.json()
        
        if data.get("status") == "success":
            analysis = data.get("analysis", [])
            # Форматируем результат
            result = "Impact Analysis:\n"
            for item in analysis:
                result += f"- {item.get('difference', 'Unknown')}: "
                result += f"{'CRITICAL' if item.get('is_critical') else 'NON-CRITICAL'} - "
                result += f"{item.get('impact_description', 'No description')}\n"
            return result
        else:
            return f"Error analyzing impact: {data.get('error', 'Unknown error')}"
    
    except requests.exceptions.RequestException as e:
        print(f"[TOOL ERROR] Failed to connect to Legal Server: {e}")
        # Fallback на mock-ответ
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
# LLM Call Node
# ============================================================================

def get_tool_schema() -> str:
    """Генерирует схему инструментов для промпта LLM."""
    schema = "TOOLS:\n"
    schema += "1. document_server.load_document(path: str) -> str\n"
    schema += "2. legal_server.compare_chunks(old_text: str, new_text: str) -> str\n"
    schema += "3. legal_server.analyze_impact(differences: str) -> str\n"
    return schema

def get_react_prompt(state: ReActState) -> str:
    """Формирует промпт для LLM."""
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
    """LLM узел - принимает решение."""
    print("\n[LLM] Calling LLM for decision...")
    
    prompt = get_react_prompt(state)
    
    # Mock-ответ LLM для демонстрации логики
    if not state["chat_history"]:
        response = """Thought: Пользователь хочет сравнить два документа. Сначала мне нужно загрузить первый документ.
Action: document_server.load_document(path="contract_old.pdf")"""
    elif "Document A" in state["observation"] and "Document B" not in state["observation"]:
        response = """Thought: Первый документ загружен. Теперь нужно загрузить второй документ.
Action: document_server.load_document(path="contract_new.pdf")"""
    elif "Document B" in state["observation"] and "differences" not in state["observation"]:
        response = """Thought: Оба документа загружены. Теперь сравню их.
Action: legal_server.compare_chunks(old_text="Document A: Contract text (old version). Clause 1: Price is $100. Clause 2: Delivery in 30 days.", new_text="Document B: Contract text (new version). Clause 1: Price is $120. Clause 2: Delivery in 15 days.")"""
    elif "differences" in state["observation"] and "critical financial impact" not in state["observation"]:
        response = """Thought: Различия найдены. Теперь проанализирую их юридическую значимость.
Action: legal_server.analyze_impact(differences='{"differences": [{"type": "MODIFIED", "detail": "Price changed from $100 to $120."}, {"type": "MODIFIED", "detail": "Delivery time reduced from 30 to 15 days."}]}')"""
    else:
        response = f"""Thought: Анализ завершен. Я могу предоставить финальный ответ.
Final Answer: В результате сравнения обнаружены следующие изменения: {state["observation"]}"""

    # Парсинг ответа
    thought = ""
    action = {}
    final_answer = ""
    
    if "Thought:" in response:
        thought = response.split("Thought:")[1].split("Action:")[0].split("Final Answer:")[0].strip()
    
    if "Action:" in response:
        action_str = response.split("Action:")[1].strip()
        tool_name = action_str.split("(")[0]
        params_str = action_str.split("(")[1].split(")")[0]
        
        params = {}
        for part in params_str.split(','):
            if '=' in part:
                key, value = part.split('=', 1)
                params[key.strip()] = value.strip().strip('"').strip("'")
        
        action = {"tool": tool_name, "params": params}
    
    if "Final Answer:" in response:
        final_answer = response.split("Final Answer:")[1].strip()
    
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
# Tool Call Node
# ============================================================================

def call_tool(state: ReActState) -> ReActState:
    """Вызывает выбранный инструмент."""
    action = state["planned_action"]
    tool_name = action["tool"]
    params = action["params"]
    
    if tool_name not in TOOLS:
        raise ValueError(f"Unknown tool: {tool_name}")
    
    tool_func = TOOLS[tool_name]
    
    try:
        observation = tool_func(**params)
        new_history = state["chat_history"] + [f"Observation: {observation}"]
        
        return {
            "chat_history": new_history,
            "observation": observation,
            "planned_action": {}
        }
    except Exception as e:
        error_msg = f"Tool {tool_name} failed with error: {e}"
        print(f"[TOOL ERROR] {error_msg}")
        
        new_history = state["chat_history"] + [f"Observation: Tool failed: {error_msg}"]
        
        return {
            "chat_history": new_history,
            "observation": error_msg,
            "planned_action": {}
        }

# ============================================================================
# Conditional Edge
# ============================================================================

def should_continue(state: ReActState) -> str:
    """Определяет, продолжить ли выполнение или завершить."""
    if state["is_finished"]:
        return "end"
    
    if state["planned_action"]:
        return "continue"
    
    return "llm"

# ============================================================================
# Graph Definition
# ============================================================================

def create_react_graph():
    """Создает граф ReAct-агента."""
    workflow = StateGraph(ReActState)
    
    workflow.add_node("llm", call_llm)
    workflow.add_node("tool", call_tool)
    
    workflow.set_entry_point("llm")
    
    workflow.add_conditional_edges(
        "llm",
        should_continue,
        {
            "continue": "tool",
            "end": END,
            "llm": "llm"
        }
    )
    
    workflow.add_edge("tool", "llm")
    
    return workflow.compile()

# ============================================================================
# Execution
# ============================================================================

if __name__ == "__main__":
    app = create_react_graph()
    
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
    
    print("--- Starting ReAct Agent Execution with HTTP Clients ---")
    print(f"Document Server: {MCP_DOCUMENT_SERVER_URL}")
    print(f"Legal Server: {MCP_LEGAL_SERVER_URL}")
    print("-" * 60)
    
    for step in app.stream(initial_state):
        print("-" * 60)
        for key, value in step.items():
            if key == "llm":
                print(f"NODE: {key}")
                print(f"  Thought: {value.get('current_thought')}")
                if value.get('planned_action'):
                    print(f"  Action: {value['planned_action']['tool']}({json.dumps(value['planned_action']['params'])})")
            elif key == "tool":
                print(f"NODE: {key}")
                obs = str(value.get('observation'))[:150]
                print(f"  Observation: {obs}...")
            elif key == '__end__':
                print(f"NODE: {key}")
                print(f"FINAL ANSWER: {step[key]['final_answer']}")
    
    print("--- ReAct Agent Execution Finished ---")
