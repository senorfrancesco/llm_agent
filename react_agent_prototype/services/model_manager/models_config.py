from typing import Dict, Any

# Mock-конфигурация моделей. В реальном проекте это будет загружаться из YAML.
# Используем пути к моделям, которые должны быть доступны в контейнере.

MODEL_CONFIG: Dict[str, Dict[str, Any]] = {
    "qwen-14b-llm": {
        "type": "text",
        "path": "/models/Qwen2.5-14B-Instruct-Q4_K_M.gguf",
        "n_gpu_layers": 30,  # Пример для Hybrid режима
        "context_size": 16384,
        "api_endpoint": "/v1/completions"
    },
    "qwen-vl-8b": {
        "type": "vision",
        "path": "/models/Qwen3-VL-8B-Instruct-Q4_K_M.gguf",
        "mmproj_path": "/models/mmproj-Qwen3-VL-8B-Instruct-F16.gguf",
        "n_gpu_layers": 20,
        "context_size": 16384,
        "api_endpoint": "/v1/chat/completions"
    },
    "labse-embedding": {
        "type": "embedding",
        "path": "/models/LaBSE-Q4_K_M.gguf",
        "n_gpu_layers": 10,
        "context_size": 512,
        "api_endpoint": "/v1/embeddings"
    }
}

# Текущая активная модель
ACTIVE_MODEL_ID = "none"
