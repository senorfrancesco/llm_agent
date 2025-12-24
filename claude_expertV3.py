import os
import json
import time
import re
import argparse
import operator
import gc
import base64
import hashlib
import requests
import numpy as np
import pymupdf
import subprocess
import signal
import socket
import atexit
from typing import TypedDict, List, Dict, Annotated, Any, Union, Optional
from scipy.optimize import linear_sum_assignment

os.environ["TOKENIZERS_PARALLELISM"] = "false"

try:
    from sentence_transformers import SentenceTransformer, util
    from langgraph.graph import StateGraph, END
    import torch
    import pdfplumber
    import pandas as pd
    from docx import Document
    from paddleocr import PaddleOCR
except ImportError:
    print("❌ pip install sentence-transformers langgraph scipy torch pdfplumber pandas python-docx paddleocr opencv-python-headless requests")
    exit()

from llama_cpp import Llama

# ========== КОНФИГУРАЦИЯ ==========

PATH_TO_LLM = "./models/gguf/qwen-14b/Qwen2.5-14B-Instruct-Q4_K_M.gguf"

EMBEDDER_CONFIG = {
    "compare": {
        "local": "./models/st/LaBSE",
        "online": "cointegrated/LaBSE-en-ru",
        "name": "LaBSE"
    },
    "equipment": {
        "local": "./models/st/LaBSE",
        "online": "cointegrated/LaBSE-en-ru",
        "name": "LaBSE"
    }
}

# Пути для локальной загрузки (если используется engine="qwen-vl")
PATH_TO_VL_MODEL = "./models/gguf/Qwen3-VL-8B-Q4/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
PATH_TO_VL_MMPROJ = "./models/gguf/Qwen3-VL-8B-Q4/mmproj-Qwen3-VL-8B-Instruct-F16.gguf"

# URL сервера (если используется engine="qwen-vl-server")
VL_SERVER_URL = "http://127.0.0.1:8081/v1/chat/completions"
VL_SERVER_PORT = 8081

DEFAULT_N_GPU_LAYERS = -1
DEFAULT_N_CTX = 16384  # ↑ Увеличено с 8192: модель поддерживает до 32K
MAX_PROMPT_TOKENS = DEFAULT_N_CTX - 2000

# Адаптивный батчинг
def get_batch_size():
    if GlobalConfig.device_mode == "cpu":
        return 16
    elif torch.cuda.is_available():
        try:
            gpu_mem = torch.cuda.get_device_properties(0).total_memory / 1e9
            if gpu_mem > 16:
                return 128
            elif gpu_mem > 8:
                return 64
            else:
                return 32
        except:
            return 32
    return 32

THRESHOLD_COMPARE = 0.72
THRESHOLD_EQUIPMENT = 0.5

# ========== ГЛОБАЛЬНЫЕ НАСТРОЙКИ ==========

class GlobalConfig:
    ocr_strategy = "auto"
    ocr_engine = "paddle"
    device_mode = "auto"
    gpu_layers = DEFAULT_N_GPU_LAYERS

# ========== УТИЛИТЫ УПРАВЛЕНИЯ СЕРВЕРОМ ==========

class ServerManager:
    """Автоматическое управление llama-server процессом"""
    _server_process = None
    _server_port = VL_SERVER_PORT
    _log_thread = None
    
    @classmethod
    def is_port_free(cls, port: int) -> bool:
        """Проверка свободности порта"""
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('127.0.0.1', port))
                return True
            except OSError:
                return False
    
    @classmethod
    def is_server_running(cls) -> bool:
        """Проверка работы сервера через HTTP запрос"""
        try:
            response = requests.get(
                f"http://127.0.0.1:{cls._server_port}/health",
                timeout=1
            )
            return response.status_code == 200
        except:
            return False
    
    @classmethod
    def _stream_output(cls, pipe, prefix):
        """Поток для вывода логов сервера в реальном времени"""
        try:
            for line in iter(pipe.readline, ''):
                if line:
                    print(f"   {prefix} {line.rstrip()}")
        except:
            pass
    
    @classmethod
    def start_server(cls) -> bool:
        """
        Запуск llama-server для Qwen-VL
        Returns: True если успешно запущен
        """
        if cls._server_process is not None:
            print("   ℹ️ Сервер уже запущен")
            return True
        
        # Проверяем существование файлов
        if not os.path.exists(PATH_TO_VL_MODEL):
            print(f"   ❌ Модель не найдена: {PATH_TO_VL_MODEL}")
            return False
        
        if not os.path.exists(PATH_TO_VL_MMPROJ):
            print(f"   ❌ MMProj не найден: {PATH_TO_VL_MMPROJ}")
            return False
        
        # Проверяем порт
        if not cls.is_port_free(cls._server_port):
            # Возможно сервер уже работает
            if cls.is_server_running():
                print(f"   ✅ Сервер уже работает на порту {cls._server_port}")
                return True
            else:
                print(f"   ⚠️ Порт {cls._server_port} занят другим процессом")
                return False
        
        print(f"\n{'='*60}")
        print(f"🚀 ЗАПУСК LLAMA-SERVER")
        print(f"{'='*60}")
        print(f"📦 Модель: {os.path.basename(PATH_TO_VL_MODEL)}")
        print(f"🎨 MMProj: {os.path.basename(PATH_TO_VL_MMPROJ)}")
        print(f"🔌 Порт: {cls._server_port}")
        print(f"🎮 GPU слоёв: {GlobalConfig.gpu_layers if GlobalConfig.device_mode != 'cpu' else 0}")
        print(f"{'='*60}\n")
        
        # Формируем команду запуска
        n_gpu = GlobalConfig.gpu_layers if GlobalConfig.device_mode != "cpu" else 0
        cmd = [
            "llama-server",
            "--model", PATH_TO_VL_MODEL,
            "--mmproj", PATH_TO_VL_MMPROJ,
            "--host", "127.0.0.1",
            "--port", str(cls._server_port),
            "--n-gpu-layers", str(n_gpu),
            "--ctx-size", "16384",  # ↑ Увеличено: модель обучена на 32K
            "--batch-size", "512",
            "--jinja",  # КРИТИЧНО для Qwen-VL: использует chat templates
            # НЕ используем --no-context-shift - каждая страница = новый запрос
            # НЕ ставим --n-predict - пусть API контролирует через max_tokens
            "--log-format", "text",
            "--verbose"
        ]
        
        print(f"💻 Команда: {' '.join(cmd)}\n")
        
        try:
            # Запускаем сервер как subprocess с перенаправлением вывода
            cls._server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,  # Объединяем stderr в stdout
                universal_newlines=True,
                bufsize=1  # Построчная буферизация
            )
            
            # Регистрируем автоматическую остановку при выходе
            atexit.register(cls.stop_server)
            
            # Запускаем поток для вывода логов
            import threading
            cls._log_thread = threading.Thread(
                target=cls._stream_output,
                args=(cls._server_process.stdout, "📋"),
                daemon=True
            )
            cls._log_thread.start()
            
            # Ждём запуска (до 45 секунд для VL моделей)
            print("⏳ Ожидание готовности сервера (это может занять ~30 сек)...")
            print("   (Загрузка модели + MMProj + инициализация)\n")
            
            for i in range(45):
                time.sleep(1)
                if cls.is_server_running():
                    print(f"\n{'='*60}")
                    print(f"✅ СЕРВЕР ГОТОВ! (загрузка заняла {i+1} сек)")
                    print(f"{'='*60}\n")
                    return True
                
                # Показываем прогресс каждые 5 секунд
                if (i + 1) % 5 == 0:
                    print(f"   ⏱️  Прошло {i+1} сек, загрузка продолжается...")
            
            print(f"\n{'='*60}")
            print("❌ ТАЙМАУТ! Сервер не запустился за 45 секунд")
            print(f"{'='*60}\n")
            cls.stop_server()
            return False
            
        except FileNotFoundError:
            print("\n❌ ОШИБКА: llama-server не найден в PATH")
            print("   Установите llama.cpp:")
            print("   https://github.com/ggerganov/llama.cpp/releases\n")
            return False
        except Exception as e:
            print(f"\n❌ ОШИБКА ЗАПУСКА: {e}\n")
            return False
    
    @classmethod
    def stop_server(cls):
        """Остановка llama-server"""
        if cls._server_process is None:
            return
        
        print(f"\n{'='*60}")
        print("🛑 ОСТАНОВКА LLAMA-SERVER")
        print(f"{'='*60}")
        
        try:
            # Пытаемся graceful shutdown
            print("   Отправка сигнала SIGTERM...")
            cls._server_process.terminate()
            
            # Ждём до 5 секунд
            try:
                cls._server_process.wait(timeout=5)
                print("   ✅ Сервер корректно завершён")
            except subprocess.TimeoutExpired:
                # Принудительное завершение
                print("   ⚠️  Таймаут, принудительное завершение...")
                cls._server_process.kill()
                cls._server_process.wait()
                print("   ✅ Сервер принудительно завершён")
                
        except Exception as e:
            print(f"   ❌ Ошибка остановки: {e}")
        finally:
            cls._server_process = None
            cls._log_thread = None
            print(f"{'='*60}\n")

# ========== СИСТЕМА ТАЙМИНГА ==========

class Timer:
    timings = {}
    
    @classmethod
    def start(cls, name: str):
        cls.timings[name] = {"start": time.time()}
    
    @classmethod
    def stop(cls, name: str):
        if name in cls.timings:
            cls.timings[name]["duration"] = time.time() - cls.timings[name]["start"]
    
    @classmethod
    def report(cls):
        print("\n" + "="*50)
        print("⏱️  СТАТИСТИКА ВРЕМЕНИ")
        print("="*50)
        total = 0
        for name, data in cls.timings.items():
            if "duration" in data:
                total += data["duration"]
                print(f"{name:30} {data['duration']:8.2f}s")
        print("-"*50)
        print(f"{'ИТОГО':30} {total:8.2f}s")
        print("="*50)

# ========== QWEN-VL SERVER КЛИЕНТ ==========

class QwenVLServer:
    """Клиент для работы с Qwen-VL через llama-server с кэшированием и retry логикой"""
    
    def __init__(self, url: str = VL_SERVER_URL, cache_dir: str = ".cache_ocr"):
        self.url = url
        self.session = requests.Session()
        self.session.headers.update({"Content-Type": "application/json"})
        self.cache_dir = cache_dir
        self.cache = {}  # in-memory кэш
        
        # Создаём директорию для кэша
        os.makedirs(cache_dir, exist_ok=True)
        
        # Автоматически запускаем сервер если не запущен
        if not ServerManager.is_server_running():
            if not ServerManager.start_server():
                raise RuntimeError("Не удалось запустить llama-server")
    
    def check_health(self) -> bool:
        """Проверка доступности сервера"""
        return ServerManager.is_server_running()
    
    def _get_cache_key(self, image_b64: str) -> str:
        """Генерация ключа кэша по хэшу изображения"""
        return hashlib.md5(image_b64.encode()).hexdigest()
    
    def _load_from_cache(self, cache_key: str) -> Optional[str]:
        """Загрузка из кэша"""
        # Сначала проверяем in-memory
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Затем проверяем файловый кэш
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        if os.path.exists(cache_file):
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    text = f.read()
                    self.cache[cache_key] = text  # сохраняем в memory
                    return text
            except:
                pass
        
        return None
    
    def _save_to_cache(self, cache_key: str, text: str):
        """Сохранение в кэш"""
        # Сохраняем в memory
        self.cache[cache_key] = text
        
        # Сохраняем в файл
        cache_file = os.path.join(self.cache_dir, f"{cache_key}.txt")
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                f.write(text)
        except:
            pass
    
    def process_image(self, image_b64: str, page_num: int = 0) -> str:
        """
        Обработка одного изображения с retry логикой и кэшированием
        
        Args:
            image_b64: Base64-закодированное изображение
            page_num: Номер страницы (для логирования)
        
        Returns:
            Извлечённый текст
        """
        cache_key = self._get_cache_key(image_b64)
        
        # Проверяем кэш
        cached_text = self._load_from_cache(cache_key)
        if cached_text is not None:
            print(f"   📦 Страница {page_num+1} из кэша")
            return cached_text
        
        # Формируем запрос
        payload = {
            "model": "qwen-vl",
            "messages": [{
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": "Извлеки весь текст с изображения. Сохрани структуру таблиц используя | для разделения колонок. Не добавляй ничего от себя."
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/png;base64,{image_b64}"}
                    }
                ]
            }],
            "max_tokens": 8192,  # ↑ Увеличено с 4096: страницы могут быть большими
            "temperature": 0.1
        }
        
        # Retry логика: 3 попытки с экспоненциальным backoff
        for attempt in range(3):
            try:
                response = self.session.post(
                    self.url,
                    json=payload,
                    timeout=180  # ↑ 3 минуты: VL модели медленные
                )
                
                if response.status_code == 200:
                    result = response.json()
                    
                    # Извлекаем текст из ответа
                    content = result.get('choices', [{}])[0].get('message', {}).get('content', '')
                    
                    if content:
                        # Сохраняем в кэш
                        self._save_to_cache(cache_key, content)
                        return content
                    else:
                        print(f"   ⚠️ Пустой ответ от сервера, попытка {attempt+1}/3")
                
                elif response.status_code == 500:
                    # Ошибка 500 часто = переполнение памяти
                    error_msg = response.json().get('error', {}).get('message', 'unknown')
                    print(f"   ⚠️ Ошибка сервера 500: {error_msg}, попытка {attempt+1}/3")
                    
                    # Если "failed to process image" - это проблема с памятью
                    if "failed to process image" in error_msg:
                        print(f"      💡 Подсказка: возможно изображение слишком большое или не хватает VRAM")
                        # Увеличиваем время ожидания после ошибки памяти
                        if attempt < 2:
                            wait_time = 10  # Даём серверу больше времени на очистку
                            print(f"   ⏳ Ожидание {wait_time}с для освобождения памяти...")
                            time.sleep(wait_time)
                        continue
                
                else:
                    print(f"   ⚠️ Ошибка {response.status_code}: {response.text[:100]}, попытка {attempt+1}/3")
                
                # Экспоненциальный backoff: 2, 4, 8 секунд
                if attempt < 2:
                    wait_time = 2 ** (attempt + 1)
                    print(f"   ⏳ Ожидание {wait_time}с перед повтором...")
                    time.sleep(wait_time)
                    
            except requests.Timeout:
                print(f"   ⏱️ Таймаут запроса (>180с), попытка {attempt+1}/3")
                if attempt < 2:
                    time.sleep(5)
            
            except Exception as e:
                print(f"   ❌ Ошибка: {e}, попытка {attempt+1}/3")
                if attempt < 2:
                    time.sleep(3)
        
        raise RuntimeError(f"Не удалось обработать страницу {page_num+1} после 3 попыток")
    
    def process_pdf_pages(self, pdf_path: str, page_indices: Optional[List[int]] = None) -> List[str]:
        """
        Обработка страниц PDF через Qwen-VL сервер
        
        Args:
            pdf_path: Путь к PDF файлу
            page_indices: Список индексов страниц для обработки (None = все страницы)
        
        Returns:
            Список текстов страниц
        """
        doc = pymupdf.open(pdf_path)
        total_pages = len(doc)
        
        if page_indices is None:
            page_indices = list(range(total_pages))
        
        results = []
        
        print(f"   🎬 Обработка через Qwen-VL: {len(page_indices)} страниц")
        
        for idx, page_num in enumerate(page_indices):
            print(f"   📄 Страница {page_num+1}/{total_pages}", end="\r")
            
            # Конвертируем страницу в PNG
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img_data = pix.tobytes("png")
            b64_data = base64.b64encode(img_data).decode('utf-8')
            
            # Обрабатываем через сервер
            try:
                text = self.process_image(b64_data, page_num)
                results.append(text)
            except Exception as e:
                print(f"\n   ❌ Ошибка обработки страницы {page_num+1}: {e}")
                results.append(f"[OCR Error: {e}]")
        
        print()  # Новая строка после прогресс-бара
        doc.close()
        return results

# ========== МЕНЕДЖЕР МОДЕЛЕЙ ==========

class ModelManager:
    _llm = None
    _embedders = {}
    _ocr_paddle = None
    _vl_model = None
    _vl_server = None
    _vl_load_failed = False 

    @classmethod
    def get_llm(cls):
        if cls._llm is None:
            # Выгружаем VL сервер если запущен
            if cls._vl_server is not None:
                print("⚠️ Выгружаю Qwen-VL Server для освобождения памяти под LLM...")
                ServerManager.stop_server()
                cls._vl_server = None
                time.sleep(2.0)
            
            # Выгружаем локальную VL если была загружена
            if cls._vl_model is not None:
                print("⚠️ Выгружаю локальную Qwen-VL для освобождения памяти под LLM...")
                try:
                    cls._vl_model.close()
                except:
                    pass
                cls._vl_model = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(1.0) 

            Timer.start("Загрузка LLM")
            if not os.path.exists(PATH_TO_LLM):
                raise FileNotFoundError(f"LLM not found: {PATH_TO_LLM}")
            
            n_gpu = 0
            split_mode = None
            
            if GlobalConfig.device_mode == "cpu":
                n_gpu = 0
                print("💻 Режим: CPU Only")
            else:
                n_gpu = GlobalConfig.gpu_layers
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    print(f"🚀 Режим: GPU ({device_count} устройств, слоев: {n_gpu})")
                    if device_count > 1:
                        split_mode = [1.0 / device_count] * device_count
                else:
                    print("⚠️ GPU не найден, используем CPU.")
                    n_gpu = 0

            n_threads = max(1, os.cpu_count() - 2)
            
            cls._llm = Llama(
                model_path=PATH_TO_LLM,
                n_ctx=DEFAULT_N_CTX,
                n_gpu_layers=n_gpu,
                n_batch=512,
                n_threads=n_threads,
                verbose=False,
                flash_attn=True,
                tensor_split=split_mode
            )
            Timer.stop("Загрузка LLM")
            print("✅ LLM готова")
        return cls._llm

    @classmethod
    def get_embedder(cls, task_type: str):
        config_key = "compare" if task_type == "compare" else "equipment"
        
        if config_key not in cls._embedders:
            cfg = EMBEDDER_CONFIG[config_key]
            Timer.start(f"Загрузка {cfg['name']}")
            print(f"🧠 Загрузка {cfg['name']}...")
            
            path = cfg['local'] if os.path.exists(cfg['local']) else cfg['online']
            
            device = "cpu"
            if GlobalConfig.device_mode != "cpu" and torch.cuda.is_available():
                device = "cuda"
            
            cls._embedders[config_key] = SentenceTransformer(path, device=device)
            Timer.stop(f"Загрузка {cfg['name']}")
            
        return cls._embedders[config_key]

    @classmethod
    def get_ocr_engine(cls):
        """Возвращает OCR движок"""
        if GlobalConfig.ocr_engine == "paddle":
            if cls._ocr_paddle is None:
                print("👁️ Инициализация PaddleOCR...")
                use_gpu = GlobalConfig.device_mode != "cpu" and torch.cuda.is_available()
                
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cls._ocr_paddle = PaddleOCR(use_angle_cls=True, lang='ru', show_log=False, use_gpu=use_gpu)
            return cls._ocr_paddle
            
        elif GlobalConfig.ocr_engine == "qwen-vl-server":
            # Выгружаем текстовую LLM чтобы освободить память
            if cls._llm is not None:
                print("⚠️ Выгружаю текстовую LLM для освобождения памяти под VL Server...")
                try:
                    cls._llm.close()
                except:
                    pass
                cls._llm = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                time.sleep(1.0)
            
            # Создаём или возвращаем QwenVLServer клиент
            if cls._vl_server is None:
                try:
                    print("🎬 Инициализация Qwen-VL Server...")
                    cls._vl_server = QwenVLServer()
                    print("✅ Qwen-VL Server готов")
                except Exception as e:
                    print(f"❌ Не удалось запустить Qwen-VL Server: {e}")
                    print("   Переключаюсь на PaddleOCR...")
                    GlobalConfig.ocr_engine = "paddle"
                    return cls.get_ocr_engine()
            
            return cls._vl_server

        elif GlobalConfig.ocr_engine == "qwen-vl":
            # Локальная загрузка (старый метод, работает нестабильно)
            if cls._vl_load_failed:
                GlobalConfig.ocr_engine = "paddle"
                return cls.get_ocr_engine()

            try:
                return cls.get_vl_model()
            except:
                GlobalConfig.ocr_engine = "paddle"
                return cls.get_ocr_engine()
        
        return None

    @classmethod
    def get_vl_model(cls):
        """Загружает локальную Qwen-VL (ВЫГРУЖАЯ LLM)"""
        if cls._vl_load_failed:
             raise RuntimeError("Qwen-VL marked as failed.")

        if cls._vl_model is None:
            if not PATH_TO_VL_MODEL or not PATH_TO_VL_MMPROJ:
                raise ValueError("Не заданы пути к Qwen-VL")
            
            if cls._llm is not None:
                print("⚠️ Выгружаю текстовую LLM для освобождения памяти под VL...")
                try: cls._llm.close()
                except: pass
                cls._llm = None
                gc.collect()
                if torch.cuda.is_available(): torch.cuda.empty_cache()
                time.sleep(2.0)
            
            print("👁️ Загрузка Qwen-VL (займет ~15 сек)...")
            n_gpu = GlobalConfig.gpu_layers if GlobalConfig.device_mode != "cpu" else 0
            
            try:
                cls._vl_model = Llama(
                    model_path=PATH_TO_VL_MODEL,
                    clip_model_path=PATH_TO_VL_MMPROJ,
                    n_ctx=8192,
                    n_gpu_layers=n_gpu,
                    verbose=False
                )
                print("✅ Qwen-VL готова")
            except Exception as e:
                print(f"❌ CRITICAL ERROR loading Qwen-VL: {e}")
                print("⚠️ Переключаюсь на PaddleOCR...")
                cls._vl_model = None
                cls._vl_load_failed = True
                GlobalConfig.ocr_engine = "paddle"
                raise e

        return cls._vl_model

    @classmethod
    def unload_all(cls):
        print("🧹 Освобождение ресурсов...")
        
        # Останавливаем VL сервер
        if cls._vl_server is not None:
            ServerManager.stop_server()
            cls._vl_server = None
        
        # Выгружаем LLM
        if cls._llm:
            try: cls._llm.close()
            except: pass
            cls._llm = None
        
        cls._embedders = {}
        cls._ocr_paddle = None
        
        # Выгружаем локальную VL модель
        if cls._vl_model:
            try: cls._vl_model.close()
            except: pass
            cls._vl_model = None
            
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ========== ЗАГРУЗЧИК ДОКУМЕНТОВ ==========

class ExtractionLog:
    """Логирование методов извлечения для прозрачности"""
    logs = []
    
    @classmethod
    def add(cls, page_num: int, method: str, items_found: int):
        cls.logs.append({
            "page": page_num,
            "method": method,
            "items": items_found
        })
        print(f"   [Стр {page_num}] 📊 {method}: найдено {items_found}")
    
    @classmethod
    def summary(cls):
        if not cls.logs:
            return ""
        report = "\n## 📈 Методы извлечения данных\n\n"
        for log in cls.logs:
            report += f"- Страница {log['page']}: {log['method']} → {log['items']} позиций\n"
        return report

class DocumentLoader:
    @staticmethod
    def read_file(path: str) -> List[str]:
        ext = path.lower().split('.')[-1]
        
        if ext == 'pdf':
            return DocumentLoader._read_pdf_smart(path)
        elif ext in ['docx', 'doc']:
            return [DocumentLoader._read_docx(path)]
        elif ext in ['jpg', 'jpeg', 'png', 'bmp', 'tiff']:
            return [DocumentLoader._process_image(path)]
        else:
            return [DocumentLoader._read_text(path)]

    @staticmethod
    def _calculate_extraction_quality(text: str, has_tables: bool) -> float:
        """
        Эвристическая оценка качества извлечения текста
        
        Returns:
            Оценка от 0 до 1 (чем выше, тем лучше)
        """
        if not text:
            return 0.0
        
        score = 0.0
        
        # 1. Длина текста (больше = лучше)
        text_length = len(text.strip())
        score += min(text_length / 500, 1.0) * 0.3
        
        # 2. Наличие таблиц (бонус)
        if has_tables:
            score += 0.2
        
        # 3. Соотношение букв к небуквенным символам
        letters = sum(c.isalpha() for c in text)
        if text_length > 0:
            letter_ratio = letters / text_length
            score += letter_ratio * 0.3
        
        # 4. Наличие слов (не артефакты)
        words = re.findall(r'\b[а-яА-ЯёЁa-zA-Z]{3,}\b', text)
        if words:
            score += min(len(words) / 50, 1.0) * 0.2
        
        return min(score, 1.0)

    @staticmethod
    def _read_pdf_smart(path: str) -> List[str]:
        """Гибридный парсер с умной склейкой таблиц + OCR"""
        pages_text = []
        quality_scores = []
        
        try:
            doc = pymupdf.open(path)
            if doc.is_encrypted:
                print(f"❌ PDF защищён: {path}")
                return []
            
            total_pages = len(doc)
            
            # Фаза 1: Стандартное извлечение с оценкой качества
            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    print(f"   📄 Анализ страницы {i+1}/{total_pages}", end="\r")
                    
                    page_content = ""
                    
                    # A. Умная обработка таблиц
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            cleaned_df_str = DocumentLoader._preprocess_table_to_string(table)
                            page_content += cleaned_df_str + "\n\n"
                    
                    # B. Текст
                    text = page.extract_text()
                    if text:
                        page_content += text + "\n"
                    
                    # Оценка качества
                    quality = DocumentLoader._calculate_extraction_quality(
                        page_content, 
                        has_tables=bool(tables)
                    )
                    
                    pages_text.append(page_content)
                    quality_scores.append(quality)
            
            print()  # Новая строка после прогресс-бара
            
            # Фаза 2: Решение об использовании OCR
            avg_quality = sum(quality_scores) / len(quality_scores) if quality_scores else 0
            
            needs_ocr = False
            ocr_reason = ""
            
            if GlobalConfig.ocr_strategy == "force":
                needs_ocr = True
                ocr_reason = "принудительный режим"
            elif GlobalConfig.ocr_strategy == "auto":
                if avg_quality < 0.3:
                    needs_ocr = True
                    ocr_reason = f"низкое качество извлечения ({avg_quality:.2f})"
                elif any(score < 0.2 for score in quality_scores):
                    needs_ocr = True
                    ocr_reason = "проблемные страницы обнаружены"
            
            # Фаза 3: OCR если необходимо
            if needs_ocr and GlobalConfig.ocr_strategy != "skip":
                print(f"   🔍 Причина OCR: {ocr_reason}")
                
                if GlobalConfig.ocr_engine == "qwen-vl-server":
                    # Используем Qwen-VL Server для обработки
                    vl_server = ModelManager.get_ocr_engine()
                    
                    if isinstance(vl_server, QwenVLServer):
                        # Определяем страницы для OCR
                        if GlobalConfig.ocr_strategy == "force":
                            pages_to_ocr = list(range(total_pages))
                        else:
                            pages_to_ocr = [i for i, score in enumerate(quality_scores) if score < 0.3]
                        
                        # Обрабатываем через сервер с адаптивным DPI
                        ocr_results = vl_server.process_pdf_pages_adaptive(path, pages_to_ocr)
                        
                        # Заменяем или дополняем
                        for idx, page_idx in enumerate(pages_to_ocr):
                            if ocr_results[idx]:
                                pages_text[page_idx] = ocr_results[idx]
                
                elif GlobalConfig.ocr_engine == "paddle":
                    # Используем PaddleOCR постранично
                    for i in range(total_pages):
                        if GlobalConfig.ocr_strategy == "force" or quality_scores[i] < 0.3:
                            print(f"   📤 OCR страницы {i+1}/{total_pages}", end="\r")
                            ocr_text = DocumentLoader._ocr_page_via_pymupdf(path, i)
                            if ocr_text:
                                pages_text[i] = ocr_text
                    print()
            
            doc.close()
            
        except Exception as e:
            print(f"❌ Ошибка чтения PDF: {e}")
            return []
            
        return [p for p in pages_text if p.strip()]

    @staticmethod
    def _preprocess_table_to_string(table: List[List[Any]]) -> str:
        """Умная склейка таблицы с сохранением иерархии"""
        if not table: 
            return ""

        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip().replace('\n', ' ') if cell is not None else "" for cell in row]
            if any(cleaned_row):
                cleaned_table.append(cleaned_row)
        
        if not cleaned_table:
            return ""

        # Определяем начало новых позиций
        text_lines = []
        current_item_number = None
        
        for row in cleaned_table:
            first_col = row[0] if row else ""
            
            # Признаки новой позиции
            is_new_item = any([
                first_col.isdigit() and len(first_col) <= 3,
                first_col.lower() == "№",
                "наименование" in "".join(row).lower() and "характеристик" in "".join(row).lower()
            ])
            
            # Признаки подхарактеристики
            is_sub_characteristic = (
                not is_new_item and 
                not first_col.isdigit() and 
                first_col != "№" and
                len(first_col) > 0
            )
            
            if is_new_item or (first_col.isdigit() and len(first_col) <= 3):
                if current_item_number and current_item_number != first_col:
                    text_lines.append("---ITEM_SEPARATOR---")
                current_item_number = first_col
                text_lines.append("[ITEM] " + " | ".join([c for c in row if c]))
            elif is_sub_characteristic:
                text_lines.append("[SPEC] " + " | ".join([c for c in row if c]))
            else:
                line_parts = [cell for cell in row if cell]
                if line_parts:
                    text_lines.append(" | ".join(line_parts))
        
        return "\n".join(text_lines)

    @staticmethod
    def _ocr_page_via_pymupdf(pdf_path, page_num):
        try:
            doc = pymupdf.open(pdf_path)
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            img_data = pix.tobytes("png")
            
            tmp_path = f"temp_ocr_{page_num}_{int(time.time())}.png"
            with open(tmp_path, "wb") as f:
                f.write(img_data)
                
            text = DocumentLoader._process_image(tmp_path)
            
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            
            doc.close()
            return text
        except Exception as e:
            return f"[OCR Error: {e}]"

    @staticmethod
    def _process_image(path: str) -> str:
        engine = GlobalConfig.ocr_engine
        
        if engine == "paddle":
            ocr = ModelManager.get_ocr_engine()
            try:
                res = ocr.ocr(path, cls=True)
                if not res or not res[0]: return ""
                return "\n".join([line[1][0] for line in res[0]])
            except Exception as e:
                return f"[PaddleOCR Error: {e}]"
            
        elif engine == "qwen-vl-server":
            try:
                vl_server = ModelManager.get_ocr_engine()
                
                if isinstance(vl_server, QwenVLServer):
                    # Читаем изображение
                    with open(path, "rb") as f:
                        img_data = f.read()
                    
                    b64_image = base64.b64encode(img_data).decode('utf-8')
                    return vl_server.process_image(b64_image, 0)
                else:
                    return "[Server not available]"
                    
            except Exception as e:
                print(f"   ⚠️ Ошибка сервера: {e}. Пробую Paddle...")
                # Fallback на Paddle
                GlobalConfig.ocr_engine = "paddle"
                return DocumentLoader._process_image(path)

        elif engine == "qwen-vl":
            try:
                vl = ModelManager.get_vl_model()
                prompt = f"<|im_start|>user\n<image>{path}</image>\nПрочитай весь текст с изображения и верни его как есть.<|im_end|>\n<|im_start|>assistant\n"
                out = vl(prompt, max_tokens=2048)
                return out['choices'][0]['text']
            except Exception as e:
                return f"[Qwen-VL Error: {e}]"
        
        return ""

    @staticmethod
    def _read_docx(path: str) -> str:
        try:
            doc = Document(path)
            text = [p.text for p in doc.paragraphs]
            for t in doc.tables:
                for r in t.rows:
                    text.append(" | ".join([c.text for c in r.cells]))
            return "\n".join(text)
        except Exception as e:
            return f"[DOCX Error: {e}]"

    @staticmethod
    def _read_text(path: str) -> str:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()

# ========== УТИЛИТЫ ==========

def parse_json_garbage(text: str) -> Union[Dict, List, None]:
    """Агрессивный парсер JSON с множественными попытками"""
    if not text:
        return None
    
    # Попытка 1: Удаляем markdown и пробуем парсить
    try:
        clean = re.sub(r'```json\s*|\s*```', '', text).strip()
        result = json.loads(clean)
        # Если получили объект вместо массива - оборачиваем в массив
        if isinstance(result, dict):
            return [result]
        return result
    except:
        pass
    
    # Попытка 2: Ищем JSON блок в тексте
    try:
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match:
            result = json.loads(match.group())
            if isinstance(result, dict):
                return [result]
            return result
    except:
        pass
    
    # Попытка 3: Ищем JSON после двоеточия
    try:
        if ':' in text:
            after_colon = text.split(':', 1)[1].strip()
            match = re.search(r'(\{.*\}|\[.*\])', after_colon, re.DOTALL)
            if match:
                result = json.loads(match.group())
                if isinstance(result, dict):
                    return [result]
                return result
    except:
        pass
    
    # Попытка 4: Парсим множественные объекты через запятую
    try:
        # Ищем паттерн: {...}, {...}, {...}
        objects = re.findall(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
        if objects:
            parsed = []
            for obj_str in objects:
                try:
                    parsed.append(json.loads(obj_str))
                except:
                    continue
            if parsed:
                return parsed
    except:
        pass
    
    # Попытка 5: Убираем всё до первой [ или {
    try:
        for start_char in ['{', '[']:
            if start_char in text:
                idx = text.index(start_char)
                json_like = text[idx:]
                end_char = '}' if start_char == '{' else ']'
                if end_char in json_like:
                    last_idx = json_like.rfind(end_char)
                    candidate = json_like[:last_idx + 1]
                    result = json.loads(candidate)
                    if isinstance(result, dict):
                        return [result]
                    return result
    except:
        pass
    
    return None

def truncate_text(text: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    max_chars = max_tokens * 4
    if len(text) <= max_chars:
        return text
    return text[:max_chars].rsplit(' ', 1)[0] + "..."

# ========== STATE ==========

class UnifiedState(TypedDict):
    task_type: str
    input_1: str
    input_2: str
    chunks_old: List[str]
    chunks_new: List[str]
    requirements: List[Dict]
    offers: List[Dict]
    matches: List[Dict]
    analysis_results: List[Any]
    final_report: str
    logs: Annotated[List[str], operator.add]

# ========== ВЕТКА COMPARE ==========

def dc_smart_chunk(text: str) -> List[str]:
    chunks = []
    section_pattern = r'\n(?=\d+\.(?:\d+\.)*\s+[А-ЯA])'
    sections = re.split(section_pattern, text)
    for section in sections:
        if len(section) > 2000:
            parts = re.split(r'\n\s*\n', section)
            for p in parts:
                clean = ' '.join(p.split())
                if len(clean) > 40: chunks.append(clean)
        else:
            clean = ' '.join(section.split())
            if len(clean) > 40: chunks.append(clean)
    return chunks

def dc_create_prompt(old, new):
    return f"""<|im_start|>system
Ты эксперт-юрист.<|im_end|>
<|im_start|>user
Сравни тексты.
СТАРЫЙ: {truncate_text(old, 2000)}
НОВЫЙ: {truncate_text(new, 2000)}
Найди юридические изменения (сроки, права, обязанности, штрафы). Игнорируй стиль.
Ответ JSON: {{"is_critical": true/false, "diff": "описание изменения", "impact": "последствия"}}<|im_end|>
<|im_start|>assistant
"""

def dc_load_node(state):
    Timer.start("dc_load")
    print("📂 [1/5] Загрузка...")
    def read(p):
        if not os.path.exists(p): return ""
        pages = DocumentLoader.read_file(p)
        return "\n".join(pages)
    
    c1 = dc_smart_chunk(read(state["input_1"]))
    c2 = dc_smart_chunk(read(state["input_2"]))
    print(f"   Чанки: {len(c1)} vs {len(c2)}")
    Timer.stop("dc_load")
    return {"chunks_old": c1, "chunks_new": c2}

def dc_match_node(state):
    Timer.start("dc_match")
    print("🧠 [2/5] Сопоставление...")
    c_old, c_new = state["chunks_old"], state["chunks_new"]
    if not c_old or not c_new: return {"matches": []}
    
    embedder = ModelManager.get_embedder("compare")
    batch_size = get_batch_size()
    
    emb_old = embedder.encode(c_old, batch_size=batch_size, show_progress_bar=True)
    emb_new = embedder.encode(c_new, batch_size=batch_size, show_progress_bar=True)
    
    scores = util.cos_sim(emb_old, emb_new).numpy()
    row_ind, col_ind = linear_sum_assignment(1 - scores)
    
    matches = []
    matched_new = set()
    
    for r, c in zip(row_ind, col_ind):
        s = float(scores[r, c])
        if s >= 0.99: 
            matched_new.add(c)
            continue
        if s >= THRESHOLD_COMPARE:
            matches.append({"type": "MODIFIED", "score": s, "old": c_old[r], "new": c_new[c]})
            matched_new.add(c)
        else:
            matches.append({"type": "DELETED", "score": 0, "old": c_old[r], "new": ""})
            
    for j, txt in enumerate(c_new):
        if j not in matched_new:
            matches.append({"type": "ADDED", "score": 0, "old": "", "new": txt})
    
    print(f"   Изменений: {len(matches)}")
    Timer.stop("dc_match")
    return {"matches": matches}

def dc_analyze_node(state):
    Timer.start("dc_analyze")
    print("🤖 [3/5] Анализ...")
    matches = state["matches"]
    results = []
    critical_kw = ["обязан", "штраф", "срок", "рублей", "не вправе", "запрещено"]
    llm = ModelManager.get_llm()
    
    for i, m in enumerate(matches):
        print(f"   {i+1}/{len(matches)}", end="\r")
        if m['type'] in ['ADDED', 'DELETED']:
            results.append({"type": m['type'], "diff": "Структурное изменение", "content": m['new'] or m['old']})
            continue
            
        txt = (m['old'] + m['new']).lower()
        if m['score'] < 0.9 or any(k in txt for k in critical_kw):
            prompt = dc_create_prompt(m['old'], m['new'])
            out = llm(prompt, max_tokens=300, temperature=0.1, echo=False)
            data = parse_json_garbage(out['choices'][0]['text'])
            
            if data and (data.get("is_critical") or len(data.get("diff","")) > 5):
                results.append({
                    "type": "MODIFIED", 
                    "critical": data.get("is_critical"),
                    "diff": data.get("diff"),
                    "impact": data.get("impact"),
                    "old": m['old'], "new": m['new']
                })
    print()
    Timer.stop("dc_analyze")
    return {"analysis_results": results}

def dc_report_node(state):
    Timer.start("dc_report")
    print("📝 [4/5] Отчет...")
    res = state["analysis_results"]
    report = "# Отчет о сравнении документов\n\n"
    report += f"**Дата:** {time.strftime('%Y-%m-%d %H:%M')}\n"
    report += f"**Найдено изменений:** {len(res)}\n\n"
    
    for r in res:
        if r['type'] == 'MODIFIED':
            icon = "⚠️" if r.get('critical') else "📝"
            report += f"### {icon} Изменено\n**Суть:** {r.get('diff')}\n**Влияние:** {r.get('impact')}\n"
            report += f"> Было: {r['old'][:150]}...\n> Стало: {r['new'][:150]}...\n\n"
        elif r['type'] == 'ADDED':
            report += f"### ✅ Добавлено\n> {r['content'][:150]}...\n\n"
        elif r['type'] == 'DELETED':
            report += f"### ❌ Удалено\n> {r['content'][:150]}...\n\n"
    
    report += ExtractionLog.summary()
    Timer.stop("dc_report")
    return {"final_report": report}

# ========== ВЕТКА EQUIPMENT ==========

def _group_fragmented_requirements(reqs: List[Dict]) -> List[Dict]:
    """Группирует разбитые характеристики одного товара"""
    if not reqs:
        return []
    
    grouped = []
    current_item = None
    merged_count = 0
    
    for req in reqs:
        item_name = req.get("item", "").strip()
        specs = req.get("specs", "").strip()
        
        # Признаки главной позиции
        is_main_item = any([
            "сервер" in item_name.lower() and len(item_name) > 15,
            "система хранения" in item_name.lower(),
            "системный блок" in item_name.lower(),
            "ноутбук" in item_name.lower(),
            "монитор" in item_name.lower(),
            "клавиатура" in item_name.lower(),
            "док-станция" in item_name.lower(),
            "или эквивалент" in item_name.lower(),
            re.match(r'^[A-Z]{2,}.*\d', item_name)
        ])
        
        # Признаки подхарактеристики
        is_sub_spec = any([
            item_name.lower().startswith(("количество", "тип", "объем", "частота", 
                                          "форм-фактор", "поддержка", "интерфейс",
                                          "наличие", "максимальное", "технология",
                                          "общий объем", "разъем", "порт", "гарантия",
                                          "установленные")),
            len(item_name) < 50 and not is_main_item,
            "не менее" in specs or "не более" in specs,
            item_name.lower().startswith("кол-во")
        ])
        
        if is_main_item or (not is_sub_spec and not current_item):
            if current_item:
                grouped.append(current_item)
                if merged_count > 0:
                    print(f"      ✓ Сгруппировано: '{current_item['item'][:50]}...' (+{merged_count} характеристик)")
                merged_count = 0
            current_item = {"item": item_name, "specs": specs}
        elif is_sub_spec and current_item:
            if current_item["specs"]:
                current_item["specs"] += f"; {item_name}: {specs}"
            else:
                current_item["specs"] = f"{item_name}: {specs}"
            merged_count += 1
        else:
            if current_item:
                grouped.append(current_item)
                if merged_count > 0:
                    print(f"      ✓ Сгруппировано: '{current_item['item'][:50]}...' (+{merged_count} характеристик)")
                merged_count = 0
            current_item = req
    
    if current_item:
        grouped.append(current_item)
        if merged_count > 0:
            print(f"      ✓ Сгруппировано: '{current_item['item'][:50]}...' (+{merged_count} характеристик)")
    
    reduction = len(reqs) - len(grouped)
    if reduction > 0:
        print(f"   📦 Объединено {reduction} подхарактеристик в главные позиции")
    
    return grouped

def eq_extract_node(state):
    Timer.start("eq_extract")
    print("📊 [1/4] Извлечение...")
    
    ExtractionLog.logs = []
    
    pages_spec = DocumentLoader.read_file(state["input_1"])
    pages_offer = DocumentLoader.read_file(state["input_2"])
    
    llm = ModelManager.get_llm()
    all_reqs = []
    all_offers = []
    
    # 1. ТЗ с умной группировкой
    print(f"   ТЗ ({len(pages_spec)} стр)...")
    for i, page in enumerate(pages_spec):
        safe_page = truncate_text(page)
        prompt = f"""<|im_start|>system
Ты анализируешь техническое задание на закупку оборудования.

КРИТИЧЕСКИ ВАЖНО - ПРАВИЛА ГРУППИРОВКИ:
1. Строки с [ITEM] - это названия позиций оборудования
2. Строки с [SPEC] сразу после [ITEM] - характеристики этой же позиции
3. Разделитель "---ITEM_SEPARATOR---" = конец одной позиции
4. ОБЪЕДИНИ все [SPEC] одной позиции в поле specs

Пример ПРАВИЛЬНОГО извлечения:
```
[ITEM] 1 | Сервер DELL PowerEdge R450
[SPEC] Тип корпуса — Rack 19"
[SPEC] Монтажная высота - Не более 1 | Юнит
[SPEC] Количество блоков питания - Не менее 2 | Шт.
---ITEM_SEPARATOR---
[ITEM] 2 | Монитор Philips 328B1
```

Правильный JSON:
[
  {{
    "item": "Сервер DELL PowerEdge R450 4LFF или эквивалент",
    "specs": "Тип корпуса: Rack 19\", Монтажная высота: не более 1 юнит, Количество блоков питания: не менее 2 шт"
  }},
  {{
    "item": "Монитор Philips 328B1 или эквивалент",
    "specs": "..."
  }}
]

НЕПРАВИЛЬНО (НЕ ДЕЛАЙ ТАК!):
[
  {{"item": "Тип корпуса"}},  ❌
  {{"item": "Монтажная высота"}}  ❌
]

ОТВЕТЬ ТОЛЬКО JSON МАССИВОМ, БЕЗ ТЕКСТА ДО И ПОСЛЕ!
Формат: [{{"item": "...", "specs": "..."}}]<|im_end|>
<|im_start|>user
{safe_page}<|im_end|>
<|im_start|>assistant
["""
        out = llm(prompt, max_tokens=2048, temperature=0.3, echo=False)  # ↑ temperature 0.1→0.3
        res = parse_json_garbage(out['choices'][0]['text'])
        if isinstance(res, list) and len(res) > 0: 
            all_reqs.extend(res)
            ExtractionLog.add(i+1, "LLM_extract_TZ", len(res))
        else:
            ExtractionLog.add(i+1, "LLM_failed", 0)
            # Дебаг: выводим что вернула LLM
            print(f"      ⚠️ Страница {i+1} вернула: {out['choices'][0]['text'][:200]}...")
    print()

    # 2. Смета
    print(f"   Смета ({len(pages_offer)} стр)...")
    for i, page in enumerate(pages_offer):
        safe_page = truncate_text(page)
        prompt = f"""<|im_start|>system
Извлеки товары из сметы/коммерческого предложения.

ВАЖНО: Один товар = один объект. Если указана цена/артикул - это отдельная позиция.

ОТВЕТЬ ТОЛЬКО JSON МАССИВОМ, БЕЗ ТЕКСТА!
Формат: [{{"name": "Полное название", "price": "Цена с ед.", "specs": "Все характеристики"}}]<|im_end|>
<|im_start|>user
{safe_page}<|im_end|>
<|im_start|>assistant
["""
        out = llm(prompt, max_tokens=2048, temperature=0.3, echo=False)  # ↑ temperature
        res = parse_json_garbage(out['choices'][0]['text'])
        if isinstance(res, list) and len(res) > 0: 
            all_offers.extend(res)
            ExtractionLog.add(i+1, "LLM_extract_Smeta", len(res))
        else:
            ExtractionLog.add(i+1, "LLM_failed", 0)
            print(f"      ⚠️ Страница {i+1} вернула: {out['choices'][0]['text'][:200]}...")
    print()

    # Пост-обработка для группировки
    print("   🔧 Группировка характеристик...")
    print(f"   До группировки: {len(all_reqs)} элементов")
    
    if all_reqs:
        print("   🔍 Примеры до группировки:")
        for i, req in enumerate(all_reqs[:5]):
            item_short = req.get('item', '')[:60]
            print(f"      [{i+1}] {item_short}")
    
    all_reqs = _group_fragmented_requirements(all_reqs)
    
    print(f"   После группировки: {len(all_reqs)} элементов")
    if all_reqs:
        print("   🔍 Примеры после группировки:")
        for i, req in enumerate(all_reqs[:3]):
            item_short = req.get('item', '')[:60]
            specs_short = req.get('specs', '')[:80]
            print(f"      [{i+1}] {item_short}")
            print(f"           Specs: {specs_short}...")
    
    # Дедупликация
    all_reqs = list({json.dumps(r, sort_keys=True): r for r in all_reqs}.values())
    all_offers = list({json.dumps(o, sort_keys=True): o for o in all_offers}.values())

    print(f"   ✅ Требований={len(all_reqs)}, Товаров={len(all_offers)}")
    Timer.stop("eq_extract")
    return {"requirements": all_reqs, "offers": all_offers}

def eq_match_node(state):
    Timer.start("eq_match")
    print("🧠 [2/4] Сопоставление...")
    reqs, offers = state["requirements"], state["offers"]
    if not reqs or not offers: return {"matches": []}
    
    embedder = ModelManager.get_embedder("equipment")
    batch_size = get_batch_size()
    
    req_txt = [f"{r.get('item','')} {r.get('specs','')}" for r in reqs]
    off_txt = [f"{o.get('name','')} {o.get('specs','')}" for o in offers]
    
    emb_req = embedder.encode(req_txt, batch_size=batch_size, show_progress_bar=False)
    emb_off = embedder.encode(off_txt, batch_size=batch_size, show_progress_bar=False)
    
    scores = util.cos_sim(emb_req, emb_off).numpy()
    
    matches = []
    for i, req in enumerate(reqs):
        best_idx = np.argmax(scores[i])
        score = scores[i][best_idx]
        
        matches.append({
            "req": req,
            "offer": offers[best_idx] if score > THRESHOLD_EQUIPMENT else None,
            "score": float(score)
        })
            
    print(f"   Сопоставлено: {sum(1 for m in matches if m['offer'])}/{len(matches)}")
    Timer.stop("eq_match")
    return {"matches": matches}

def eq_evaluate_node(state):
    Timer.start("eq_evaluate")
    print("⚖️ [3/4] Проверка...")
    matches = state["matches"]
    results = []
    llm = ModelManager.get_llm()
    
    for i, pair in enumerate(matches):
        req, offer = pair['req'], pair['offer']
        print(f"   {i+1}/{len(matches)}", end="\r")
        
        if not offer:
            results.append({"status": "MISSING", "item": req.get("item"), "reason": "Нет в смете"})
            continue
            
        prompt = f"""<|im_start|>system
Проверь соответствие. Лучше/равно - OK. Хуже - FAIL.
Ответ JSON: {{"pass": true/false, "reason": "почему"}}<|im_end|>
<|im_start|>user
ТРЕБОВАНИЕ: {json.dumps(req, ensure_ascii=False)}
ПРЕДЛОЖЕНИЕ: {json.dumps(offer, ensure_ascii=False)}<|im_end|>
<|im_start|>assistant
"""
        out = llm(prompt, max_tokens=256, temperature=0.1)
        eval_res = parse_json_garbage(out['choices'][0]['text'])
        status = "OK" if eval_res and eval_res.get("pass") else "FAIL"
        results.append({
            "status": status, "item": req.get("item"), "offer": offer.get("name"),
            "specs": offer.get("specs"), "price": offer.get("price"),
            "reason": eval_res.get("reason", "Ошибка") if eval_res else "Сбой"
        })
    print()
    Timer.stop("eq_evaluate")
    return {"analysis_results": results}

def eq_report_node(state):
    Timer.start("eq_report")
    print("📝 [4/4] Отчет...")
    res = state["analysis_results"]
    ok = sum(1 for r in res if r['status'] == 'OK')
    total = len(res)
    
    report = f"# 🛠 Анализ закупки\n**Дата:** {time.strftime('%Y-%m-%d %H:%M')}\n"
    report += f"**Соответствие:** {ok}/{total}\n\n"
    
    for r in res:
        icon = "✅" if r['status'] == "OK" else ("❓" if r['status'] == "MISSING" else "❌")
        
        item_name = r.get('item', 'Неизвестно')
        if len(item_name) > 80:
            item_name = item_name[:77] + "..."
        
        report += f"### {icon} {item_name}\n"
        
        if r['status'] != "MISSING":
            offer_name = r.get('offer', 'Не указано')
            price = r.get('price', 'Не указана')
            specs = r.get('specs', 'Не указаны')
            
            if len(specs) > 500:
                specs = specs[:497] + "..."
            
            report += f"**Товар:** {offer_name} ({price})\n"
            report += f"**Спецификация:** {specs}\n"
        
        reason = r.get('reason', 'Нет данных')
        report += f"**Вывод:** {reason}\n\n---\n"
    
    report += ExtractionLog.summary()
    Timer.stop("eq_report")
    return {"final_report": report}

# ========== ГРАФ ==========

def build_graph(task_type: str):
    workflow = StateGraph(UnifiedState)
    
    if task_type == "compare":
        workflow.add_node("dc_load", dc_load_node)
        workflow.add_node("dc_match", dc_match_node)
        workflow.add_node("dc_analyze", dc_analyze_node)
        workflow.add_node("dc_report", dc_report_node)
        
        workflow.set_entry_point("dc_load")
        workflow.add_edge("dc_load", "dc_match")
        workflow.add_edge("dc_match", "dc_analyze")
        workflow.add_edge("dc_analyze", "dc_report")
        workflow.add_edge("dc_report", END)
        
    else:
        workflow.add_node("eq_extract", eq_extract_node)
        workflow.add_node("eq_match", eq_match_node)
        workflow.add_node("eq_evaluate", eq_evaluate_node)
        workflow.add_node("eq_report", eq_report_node)
        
        workflow.set_entry_point("eq_extract")
        workflow.add_edge("eq_extract", "eq_match")
        workflow.add_edge("eq_match", "eq_evaluate")
        workflow.add_edge("eq_evaluate", "eq_report")
        workflow.add_edge("eq_report", END)
    
    return workflow.compile()

# ========== ЗАПУСК ==========

def signal_handler(sig, frame):
    """Обработчик Ctrl+C для корректного завершения"""
    print("\n\n⚠️ Получен сигнал прерывания...")
    ModelManager.unload_all()
    ServerManager.stop_server()
    print("👋 Завершено")
    exit(0)

if __name__ == "__main__":
    # Регистрируем обработчик сигналов
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    parser = argparse.ArgumentParser(description="Unified AI Agent v6.0 - Auto llama-server Edition")
    parser.add_argument("--task", type=str, choices=["compare", "equipment"], required=True)
    parser.add_argument("file1", help="Файл 1 (Старый/ТЗ)")
    parser.add_argument("file2", help="Файл 2 (Новый/Смета)")
    
    parser.add_argument("--ocr-strategy", choices=["auto", "force", "skip"], default="auto",
                       help="auto=OCR при низком качестве, force=всегда, skip=никогда")
    parser.add_argument("--ocr-engine", choices=["paddle", "qwen-vl-server"], default="paddle",
                       help="paddle=PaddleOCR (быстро), qwen-vl-server=Qwen-VL автозапуск (точно)")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.file1): exit(f"❌ {args.file1}")
    if not os.path.exists(args.file2): exit(f"❌ {args.file2}")
    
    GlobalConfig.ocr_strategy = args.ocr_strategy
    GlobalConfig.ocr_engine = args.ocr_engine
    GlobalConfig.device_mode = args.device
    
    print("="*50)
    print(f"🚀 ЗАДАЧА: {args.task.upper()}")
    print(f"📄 1: {os.path.basename(args.file1)}")
    print(f"📄 2: {os.path.basename(args.file2)}")
    print(f"🔧 Config: OCR={args.ocr_strategy}/{args.ocr_engine}, Device={args.device}")
    print("="*50)
    
    try:
        Timer.start("ОБЩЕЕ ВРЕМЯ")
        app = build_graph(args.task)
        res = app.invoke({
            "task_type": args.task,
            "input_1": args.file1,
            "input_2": args.file2,
            "requirements": [], "offers": [], "matches": [], 
            "analysis_results": [], "final_report": "", "logs": [],
            "chunks_old": [], "chunks_new": []
        })
        Timer.stop("ОБЩЕЕ ВРЕМЯ")
        
        out_name = f"Report_{args.task}_{int(time.time())}.md"
        with open(out_name, "w", encoding="utf-8") as f:
            f.write(res["final_report"])
        
        print(f"\n✅ ГОТОВО: {out_name}")
        Timer.report()
        
    except Exception as e:
        print(f"\n❌ ОШИБКА: {e}")
        import traceback
        traceback.print_exc()
    finally:
        ModelManager.unload_all()
        ServerManager.stop_server()