"""
UMS Client - HTTP-клиент для общения MCP-серверов с Unified Model Server.

Все MCP-серверы используют этот клиент для запросов инференса.
В реальном проекте UMS будет отдельным процессом на порту 8090.
"""

import requests
from typing import Dict, Any, Optional
import json

UMS_URL = "http://localhost:8090"

class UMSClient:
    """HTTP-клиент для взаимодействия с UMS."""
    
    def __init__(self, base_url: str = UMS_URL):
        self.base_url = base_url
    
    def infer(self, model_id: str, payload: Dict[str, Any], mode: str = "auto") -> Dict[str, Any]:
        """
        Выполняет инференс через UMS.
        
        Args:
            model_id: ID модели (например, "qwen-14b-llm", "labse-embedding")
            payload: Тело запроса, специфичное для модели
            mode: Режим работы ("auto", "gpu", "cpu", "hybrid")
        
        Returns:
            Результат инференса
        """
        url = f"{self.base_url}/infer"
        
        request_body = {
            "model_id": model_id,
            "mode": mode,
            "priority": "normal",
            "request_body": payload
        }
        
        try:
            print(f"[UMS_CLIENT] Sending request to {url} for model {model_id}")
            response = requests.post(url, json=request_body, timeout=120)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[UMS_CLIENT] Error: {e}")
            raise RuntimeError(f"UMS inference failed: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Получает статус UMS (загруженные модели, использование памяти)."""
        url = f"{self.base_url}/status"
        
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"[UMS_CLIENT] Error getting status: {e}")
            return {"error": str(e)}

# Глобальный экземпляр клиента
ums_client = UMSClient()

# ============================================================================
# Вспомогательные функции для MCP-серверов
# ============================================================================

def generate_text_via_ums(prompt: str, max_tokens: int = 512) -> str:
    """
    Генерирует текст через UMS, используя LLM-модель.
    
    Используется MCP Legal Server для анализа.
    """
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": 0.7,
        "top_p": 0.9
    }
    
    response = ums_client.infer("qwen-14b-llm", payload)
    
    # Mock-ответ, если UMS недоступен
    if "error" in response:
        return f"[MOCK] Response for prompt: {prompt[:50]}..."
    
    return response.get("content", "")

def get_embeddings_via_ums(text: str) -> list:
    """
    Получает эмбеддинги текста через UMS, используя Embedding-модель.
    
    Используется MCP Legal Server для сравнения текстов.
    """
    payload = {
        "input": text,
        "normalize": True
    }
    
    response = ums_client.infer("labse-embedding", payload)
    
    # Mock-ответ, если UMS недоступен
    if "error" in response:
        return [0.1] * 768  # Вектор размером 768 (LaBSE)
    
    return response.get("embedding", [0.1] * 768)

def process_vision_via_ums(image_path: str, prompt: str) -> str:
    """
    Обрабатывает изображение через UMS, используя Vision-модель.
    
    Используется MCP Document Server для OCR и анализа изображений.
    """
    payload = {
        "image_path": image_path,
        "prompt": prompt
    }
    
    response = ums_client.infer("qwen-vl-8b", payload)
    
    # Mock-ответ, если UMS недоступен
    if "error" in response:
        return f"[MOCK] Vision response for: {prompt[:50]}..."
    
    return response.get("response", "")
