"""
Unified Model Server (UMS) - Реальный прототип с управлением llama-server.

Функции:
- Динамическая загрузка/выгрузка моделей через llama-cpp-python
- Поддержка режимов: GPU, CPU, Hybrid
- Smart Swapping (LRU) для оптимизации VRAM
- HTTP API для взаимодействия с MCP-серверами
"""

import os
import subprocess
import time
import requests
import json
import psutil
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict
from enum import Enum
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# ============================================================================
# Configuration & Constants
# ============================================================================

UMS_PORT = 8090
LLAMA_SERVER_PORT = 8091  # Порт для llama-server
LLAMA_SERVER_URL = f"http://localhost:{LLAMA_SERVER_PORT}"

class DeviceMode(str, Enum):
    """Режимы работы GPU/CPU."""
    GPU = "gpu"
    CPU = "cpu"
    HYBRID = "hybrid"

class LoadStrategy(str, Enum):
    """Стратегии загрузки моделей."""
    EAGER = "eager"        # Загрузить при старте
    LAZY = "lazy"          # Загрузить при первом запросе
    ON_DEMAND = "on_demand" # Загрузить на каждый запрос
    SMART = "smart"        # Автоматическая выгрузка LRU

# ============================================================================
# Model Configuration
# ============================================================================

@dataclass
class ModelConfig:
    """Конфигурация модели."""
    model_id: str
    model_type: str  # "text", "vision", "embedding"
    path: str
    n_gpu_layers: int = 0
    context_size: int = 512
    mmproj_path: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

# Конфигурация доступных моделей
MODELS_CONFIG: Dict[str, ModelConfig] = {
    "qwen-14b-llm": ModelConfig(
        model_id="qwen-14b-llm",
        model_type="text",
        path=os.getenv("MODEL_PATH_QWEN14B", "/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf"),
        n_gpu_layers=30,
        context_size=16384
    ),
    "qwen-vl-8b": ModelConfig(
        model_id="qwen-vl-8b",
        model_type="vision",
        path=os.getenv("MODEL_PATH_QWENVL", "/models/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"),
        n_gpu_layers=20,
        context_size=16384,
        mmproj_path=os.getenv("MMPROJ_PATH", "/models/mmproj-Qwen3-VL-8B-Instruct-F16.gguf")
    ),
    "labse-embedding": ModelConfig(
        model_id="labse-embedding",
        model_type="embedding",
        path=os.getenv("MODEL_PATH_LABSE", "/models/LaBSE-Q4_K_M.gguf"),
        n_gpu_layers=10,
        context_size=512
    )
}

# ============================================================================
# Global State
# ============================================================================

LLAMA_SERVER_PROCESS: Optional[subprocess.Popen] = None
ACTIVE_MODEL_ID: str = "none"
DEVICE_MODE: DeviceMode = DeviceMode.HYBRID  # По умолчанию
LOAD_STRATEGY: LoadStrategy = LoadStrategy.LAZY
MODEL_ACCESS_LOG: Dict[str, float] = {}  # Для LRU

# ============================================================================
# Helper Functions
# ============================================================================

def _get_available_vram() -> float:
    """Получает доступную VRAM в ГБ."""
    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return info.free / (1024 ** 3)
    except Exception:
        return 0.0

def _get_available_ram() -> float:
    """Получает доступную RAM в ГБ."""
    try:
        mem = psutil.virtual_memory()
        return mem.available / (1024 ** 3)
    except Exception:
        return 0.0

def _is_server_running() -> bool:
    """Проверяет, запущен ли llama-server."""
    try:
        response = requests.get(f"{LLAMA_SERVER_URL}/health", timeout=1)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def _stop_server():
    """Останавливает текущий llama-server."""
    global LLAMA_SERVER_PROCESS, ACTIVE_MODEL_ID
    
    if LLAMA_SERVER_PROCESS:
        print(f"[UMS] Stopping model: {ACTIVE_MODEL_ID}")
        try:
            LLAMA_SERVER_PROCESS.terminate()
            LLAMA_SERVER_PROCESS.wait(timeout=10)
        except subprocess.TimeoutExpired:
            print(f"[UMS] Force killing llama-server")
            LLAMA_SERVER_PROCESS.kill()
        except Exception as e:
            print(f"[UMS] Error stopping server: {e}")
        finally:
            LLAMA_SERVER_PROCESS = None
            ACTIVE_MODEL_ID = "none"

def _start_server(model_id: str, device_mode: DeviceMode = DeviceMode.HYBRID):
    """Запускает llama-server с указанной моделью."""
    global LLAMA_SERVER_PROCESS, ACTIVE_MODEL_ID
    
    config = MODELS_CONFIG.get(model_id)
    if not config:
        raise ValueError(f"Model ID '{model_id}' not found in config.")
    
    # Проверяем наличие файла модели
    if not os.path.exists(config.path):
        raise FileNotFoundError(f"Model file not found: {config.path}")
    
    # Формируем команду запуска
    cmd = [
        "python3.11", "-m", "llama_cpp.server",
        "--model", config.path,
        "--port", str(LLAMA_SERVER_PORT),
        "--host", "0.0.0.0",
        "--n_ctx", str(config.context_size),
    ]
    
    # Добавляем параметры в зависимости от режима
    if device_mode == DeviceMode.GPU:
        cmd.extend(["--n_gpu_layers", str(config.n_gpu_layers)])
    elif device_mode == DeviceMode.HYBRID:
        # Гибридный режим: часть слоёв на GPU, часть на CPU
        cmd.extend(["--n_gpu_layers", str(max(1, config.n_gpu_layers // 2))])
    # Для CPU режима не добавляем --n_gpu_layers (или устанавливаем в 0)
    
    # Добавляем multimodal projection для Vision моделей
    if config.mmproj_path and os.path.exists(config.mmproj_path):
        cmd.extend(["--mmproj", config.mmproj_path])
    
    print(f"[UMS] Starting llama-server for model: {model_id}")
    print(f"[UMS] Device mode: {device_mode}")
    print(f"[UMS] Command: {' '.join(cmd)}")
    
    try:
        LLAMA_SERVER_PROCESS = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        
        # Ожидаем готовности сервера
        start_time = time.time()
        timeout = 120
        
        while time.time() - start_time < timeout:
            if _is_server_running():
                ACTIVE_MODEL_ID = model_id
                MODEL_ACCESS_LOG[model_id] = time.time()
                print(f"[UMS] Model {model_id} loaded successfully")
                return
            time.sleep(1)
        
        _stop_server()
        raise TimeoutError(f"llama-server failed to start within {timeout}s")
    
    except Exception as e:
        _stop_server()
        raise RuntimeError(f"Failed to start llama-server: {e}")

# ============================================================================
# Core Functions
# ============================================================================

def switch_model(model_id: str, device_mode: DeviceMode = DeviceMode.HYBRID):
    """Переключает активную модель."""
    global ACTIVE_MODEL_ID
    
    if model_id == ACTIVE_MODEL_ID:
        print(f"[UMS] Model {model_id} is already active")
        return
    
    # Останавливаем текущую модель
    _stop_server()
    
    # Запускаем новую
    _start_server(model_id, device_mode)

def infer(model_id: str, payload: Dict[str, Any], device_mode: DeviceMode = DeviceMode.HYBRID) -> Dict[str, Any]:
    """
    Выполняет инференс. Если модель не активна, переключает её.
    """
    if model_id != ACTIVE_MODEL_ID:
        switch_model(model_id, device_mode)
    
    config = MODELS_CONFIG.get(model_id)
    if not config:
        raise ValueError(f"Model ID '{model_id}' not found")
    
    # Определяем endpoint в зависимости от типа модели
    if config.model_type == "text":
        endpoint = "/v1/completions"
    elif config.model_type == "vision":
        endpoint = "/v1/chat/completions"
    elif config.model_type == "embedding":
        endpoint = "/v1/embeddings"
    else:
        raise ValueError(f"Unknown model type: {config.model_type}")
    
    url = f"{LLAMA_SERVER_URL}{endpoint}"
    
    print(f"[UMS] Sending inference request to {url}")
    
    try:
        response = requests.post(url, json=payload, timeout=300)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"[UMS] Inference error: {e}")
        raise RuntimeError(f"Inference failed: {e}")

def get_status() -> Dict[str, Any]:
    """Возвращает статус UMS."""
    return {
        "active_model": ACTIVE_MODEL_ID,
        "device_mode": DEVICE_MODE.value,
        "load_strategy": LOAD_STRATEGY.value,
        "available_vram_gb": _get_available_vram(),
        "available_ram_gb": _get_available_ram(),
        "server_running": _is_server_running(),
        "models_config": {k: v.to_dict() for k, v in MODELS_CONFIG.items()},
        "model_access_log": MODEL_ACCESS_LOG
    }

# ============================================================================
# FastAPI App
# ============================================================================

app = FastAPI(title="Unified Model Server", version="1.0.0")

class InferenceRequest(BaseModel):
    model_id: str
    payload: Dict[str, Any]
    device_mode: str = "hybrid"
    priority: str = "normal"

class SwitchModelRequest(BaseModel):
    model_id: str
    device_mode: str = "hybrid"

@app.get("/health")
async def health():
    """Health check."""
    return {"status": "healthy", "service": "ums"}

@app.get("/status")
async def status():
    """Получить статус UMS."""
    return get_status()

@app.post("/infer")
async def infer_endpoint(request: InferenceRequest):
    """Выполнить инференс."""
    try:
        device_mode = DeviceMode(request.device_mode)
        result = infer(request.model_id, request.payload, device_mode)
        return {"status": "success", "result": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/switch_model")
async def switch_model_endpoint(request: SwitchModelRequest):
    """Переключить активную модель."""
    try:
        device_mode = DeviceMode(request.device_mode)
        switch_model(request.model_id, device_mode)
        return {"status": "success", "active_model": request.model_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/stop")
async def stop_server():
    """Остановить текущий llama-server."""
    try:
        _stop_server()
        return {"status": "success", "message": "Server stopped"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ============================================================================
# Cleanup on Exit
# ============================================================================

import atexit

def cleanup():
    """Очистка при завершении."""
    print("[UMS] Cleaning up...")
    _stop_server()

atexit.register(cleanup)

# ============================================================================
# Main
# ============================================================================

if __name__ == "__main__":
    print("[UMS] Starting Unified Model Server...")
    print(f"[UMS] Available models: {list(MODELS_CONFIG.keys())}")
    print(f"[UMS] Device mode: {DEVICE_MODE}")
    print(f"[UMS] Load strategy: {LOAD_STRATEGY}")
    
    uvicorn.run(app, host="0.0.0.0", port=UMS_PORT)
