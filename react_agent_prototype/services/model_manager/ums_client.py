"""
UMS Client - HTTP-клиент для взаимодействия с Unified Model Server.

Все MCP-серверы используют этот клиент для выполнения инференса.
"""

import requests
from typing import Dict, Any, Optional, List
import json

UMS_URL = "http://localhost:8090"

class UMSClient:
    """HTTP-клиент для взаимодействия с UMS."""
    
    def __init__(self, base_url: str = UMS_URL):
        self.base_url = base_url
    
    def infer(self, model_id: str, payload: Dict[str, Any], device_mode: str = "hybrid") -> Dict[str, Any]:
        """
        Выполняет инференс через UMS.
        
        Args:
            model_id: ID модели (qwen-14b-llm, qwen-vl-8b, labse-embedding)
            payload: Тело запроса, специфичное для модели
            device_mode: Режим работы (gpu, cpu, hybrid)
        
        Returns:
            Результат инференса
        """
        url = f"{self.base_url}/infer"
        
        request_body = {
            "model_id": model_id,
            "payload": payload,
            "device_mode": device_mode,
            "priority": "normal"
        }
        
        try:
            print(f"[UMS_CLIENT] Sending inference request for model: {model_id}")
            response = requests.post(url, json=request_body, timeout=300)
            response.raise_for_status()
            data = response.json()
            
            if data.get("status") == "success":
                return data.get("result", {})
            else:
                raise RuntimeError(f"UMS returned error: {data}")
        
        except requests.exceptions.RequestException as e:
            print(f"[UMS_CLIENT] Error: {e}")
            raise RuntimeError(f"Failed to connect to UMS: {e}")
    
    def switch_model(self, model_id: str, device_mode: str = "hybrid") -> Dict[str, Any]:
        """Переключает активную модель на UMS."""
        url = f"{self.base_url}/switch_model"
        
        request_body = {
            "model_id": model_id,
            "device_mode": device_mode
        }
        
        try:
            print(f"[UMS_CLIENT] Switching to model: {model_id}")
            response = requests.post(url, json=request_body, timeout=120)
            response.raise_for_status()
            return response.json()
        
        except requests.exceptions.RequestException as e:
            print(f"[UMS_CLIENT] Error switching model: {e}")
            raise RuntimeError(f"Failed to switch model: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Получает статус UMS."""
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

def generate_text_via_ums(prompt: str, max_tokens: int = 512, temperature: float = 0.7) -> str:
    """
    Генерирует текст через UMS, используя LLM-модель (Qwen-14B).
    
    Используется MCP Legal Server для анализа.
    """
    payload = {
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "top_p": 0.9
    }
    
    try:
        response = ums_client.infer("qwen-14b-llm", payload, device_mode="hybrid")
        
        # Парсим ответ от llama-server
        if "choices" in response:
            return response["choices"][0].get("text", "")
        elif "content" in response:
            return response["content"]
        else:
            return str(response)
    
    except Exception as e:
        print(f"[UMS_CLIENT] Error generating text: {e}")
        raise

def get_embeddings_via_ums(text: str, normalize: bool = True) -> List[float]:
    """
    Получает эмбеддинги текста через UMS, используя Embedding-модель (LaBSE).
    
    Используется MCP Legal Server для сравнения текстов.
    """
    payload = {
        "input": text,
        "normalize": normalize
    }
    
    try:
        response = ums_client.infer("labse-embedding", payload, device_mode="cpu")
        
        # Парсим ответ от llama-server
        if "data" in response:
            return response["data"][0].get("embedding", [])
        elif "embedding" in response:
            return response["embedding"]
        else:
            return []
    
    except Exception as e:
        print(f"[UMS_CLIENT] Error getting embeddings: {e}")
        raise

def process_vision_via_ums(image_path: str, prompt: str) -> str:
    """
    Обрабатывает изображение через UMS, используя Vision-модель (Qwen-VL).
    
    Используется MCP Document Server для OCR и анализа изображений.
    """
    payload = {
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": image_path}},
                    {"type": "text", "text": prompt}
                ]
            }
        ]
    }
    
    try:
        response = ums_client.infer("qwen-vl-8b", payload, device_mode="hybrid")
        
        # Парсим ответ от llama-server
        if "choices" in response:
            return response["choices"][0].get("message", {}).get("content", "")
        else:
            return str(response)
    
    except Exception as e:
        print(f"[UMS_CLIENT] Error processing vision: {e}")
        raise
