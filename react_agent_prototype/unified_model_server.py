import os
import time
import subprocess
import requests
from typing import Optional, Dict, Any
from models_config import MODEL_CONFIG, ACTIVE_MODEL_ID

# Глобальные переменные для управления процессом
LLAMA_SERVER_PROCESS: Optional[subprocess.Popen] = None
LLAMA_SERVER_PORT = 8080
LLAMA_SERVER_URL = f"http://localhost:{LLAMA_SERVER_PORT}"

def _is_server_running() -> bool:
    """Проверяет, запущен ли llama-server и отвечает ли он."""
    try:
        response = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=1)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def _stop_server():
    """Останавливает текущий запущенный llama-server."""
    global LLAMA_SERVER_PROCESS, ACTIVE_MODEL_ID
    if LLAMA_SERVER_PROCESS:
        print(f"[UMS] Stopping current model: {ACTIVE_MODEL_ID}")
        LLAMA_SERVER_PROCESS.terminate()
        LLAMA_SERVER_PROCESS.wait(timeout=10)
        LLAMA_SERVER_PROCESS = None
        ACTIVE_MODEL_ID = "none"
        print("[UMS] Server stopped.")

def _start_server(model_id: str):
    """Запускает llama-server с указанной моделью."""
    global LLAMA_SERVER_PROCESS, ACTIVE_MODEL_ID
    
    config = MODEL_CONFIG.get(model_id)
    if not config:
        raise ValueError(f"Model ID '{model_id}' not found in config.")

    # 1. Формирование команды
    cmd = [
        "python3.11", "-m", "llama_cpp.server",
        "--model", config["path"],
        "--port", str(LLAMA_SERVER_PORT),
        "--n_gpu_layers", str(config.get("n_gpu_layers", 0)),
        "--n_ctx", str(config.get("context_size", 512)),
        "--host", "0.0.0.0"
    ]
    
    if config.get("mmproj_path"):
        cmd.extend(["--mmproj", config["mmproj_path"]])
        
    # 2. Запуск процесса
    print(f"[UMS] Starting server for model: {model_id} on port {LLAMA_SERVER_PORT}")
    print(f"[UMS] Command: {' '.join(cmd)}")
    
    # Запускаем в отдельном процессе
    LLAMA_SERVER_PROCESS = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # 3. Ожидание готовности
    start_time = time.time()
    timeout = 60  # Даем 60 секунд на загрузку
    while time.time() - start_time < timeout:
        if _is_server_running():
            ACTIVE_MODEL_ID = model_id
            print(f"[UMS] Model {model_id} loaded and server is ready.")
            return
        time.sleep(1)
    
    _stop_server()
    raise TimeoutError(f"Server failed to start model {model_id} within {timeout} seconds.")

def switch_model(model_id: str):
    """Переключает активную модель, выгружая предыдущую, если она есть."""
    global ACTIVE_MODEL_ID
    
    if model_id == ACTIVE_MODEL_ID:
        print(f"[UMS] Model {model_id} is already active.")
        return
    
    # 1. Останавливаем текущий сервер
    _stop_server()
    
    # 2. Запускаем новый
    _start_server(model_id)

def infer(model_id: str, payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Выполняет инференс. Если модель не активна, переключает ее.
    """
    if model_id != ACTIVE_MODEL_ID:
        switch_model(model_id)
    
    config = MODEL_CONFIG.get(model_id)
    if not config:
        raise ValueError(f"Model ID '{model_id}' not found in config.")
        
    endpoint = config["api_endpoint"]
    url = f"{LLAMA_SERVER_URL}{endpoint}"
    
    print(f"[UMS] Sending request to {url} for model {model_id}")
    
    try:
        # В реальном проекте здесь будет более сложная обработка
        response = requests.post(url, json=payload, timeout=120)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[UMS] Error during inference: {e}")
        raise RuntimeError(f"Inference failed for model {model_id}: {e}")

# --- Mock-функции для демонстрации ---
def generate_text(prompt: str) -> str:
    """Имитация вызова LLM для генерации текста."""
    payload = {
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.7
    }
    # Используем LLM-модель
    response = infer("qwen-14b-llm", payload)
    # Mock-ответ, так как мы не запускаем реальный сервер
    return response.get("content", f"Mocked response for: {prompt[:50]}...")

def get_embeddings(text: str) -> list:
    """Имитация вызова Embedding модели."""
    payload = {
        "input": text,
        "model": "labse-embedding"
    }
    # Используем Embedding-модель
    response = infer("labse-embedding", payload)
    # Mock-ответ
    return response.get("data", [0.1] * 768)

def process_vision(image_path: str, prompt: str) -> str:
    """Имитация вызова Vision модели."""
    # В реальном проекте image_path будет преобразован в base64 или передан URL
    payload = {
        "messages": [
            {"role": "user", "content": f"Image: {image_path}, Prompt: {prompt}"}
        ]
    }
    # Используем Vision-модель
    response = infer("qwen-vl-8b", payload)
    # Mock-ответ
    return response.get("choices", [{}])[0].get("message", {}).get("content", f"Mocked vision response for: {prompt[:50]}...")

# --- Пример использования (для тестирования) ---
if __name__ == "__main__":
    # В реальном проекте нужно убедиться, что модели доступны по указанным путям
    # и что llama-cpp-python установлен с поддержкой сервера.
    print("--- UMS Mock Test ---")
    
    try:
        # 1. Первый вызов - загрузит qwen-14b-llm
        text_result = generate_text("Explain the ReAct pattern in one sentence.")
        print(f"Result 1 (Text): {text_result}")
        
        # 2. Второй вызов той же модели - не будет переключать
        text_result_2 = generate_text("What is the main benefit of microservices?")
        print(f"Result 2 (Text): {text_result_2}")
        
        # 3. Вызов другой модели - выгрузит LLM и загрузит Embedding
        embedding_result = get_embeddings("Test sentence for embedding.")
        print(f"Result 3 (Embedding vector size): {len(embedding_result)}")
        
        # 4. Вызов Vision модели - выгрузит Embedding и загрузит Vision
        vision_result = process_vision("/path/to/contract.pdf", "Extract the table.")
        print(f"Result 4 (Vision): {vision_result}")
        
    except Exception as e:
        print(f"An error occurred during UMS test: {e}")
    finally:
        # Обязательно останавливаем сервер при завершении
        _stop_server()
