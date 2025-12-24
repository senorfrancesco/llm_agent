import os
import json
import time
import re
import argparse
import operator
import gc
import base64
import requests
import numpy as np
import subprocess
import signal
import pymupdf  # PyMuPDF (fitz)
from typing import TypedDict, List, Dict, Annotated, Any, Union
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
    print("❌ pip install sentence-transformers langgraph scipy torch pdfplumber pandas python-docx paddleocr opencv-python-headless requests pymupdf")
    exit()

from llama_cpp import Llama

# ========== КОНФИГУРАЦИЯ ==========

# Путь к текстовой модели (аналитик)
PATH_TO_LLM = "./models/gguf/qwen-14b/Qwen2.5-14B-Instruct-Q4_K_M.gguf"

# Пути к Vision модели (для запуска сервера)
# Укажите здесь ваши реальные пути
PATH_TO_VL_MODEL = "./models/gguf/Qwen3-VL-8B-Q4/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
PATH_TO_VL_MMPROJ = "./models/gguf/Qwen3-VL-8B-Q4/mmproj-Qwen3-VL-8B-Instruct-F16.gguf"

EMBEDDER_CONFIG = {
    "compare": {
        "local": "./models/st/E5-legal",
        "online": "SergeyKarpenko1/multilingual-e5-small-legal-matryoshka_384",
        "name": "E5-Legal"
    },
    "equipment": {
        "local": "./models/st/LaBSE",
        "online": "cointegrated/LaBSE-en-ru",
        "name": "LaBSE"
    }
}

# Настройки сервера
VL_SERVER_HOST = "127.0.0.1"
VL_SERVER_PORT = "8081"
VL_SERVER_URL = f"http://{VL_SERVER_HOST}:{VL_SERVER_PORT}/v1/chat/completions"

DEFAULT_N_GPU_LAYERS = -1
DEFAULT_N_CTX = 8192
MAX_PROMPT_TOKENS = DEFAULT_N_CTX - 2000
EMBEDDING_BATCH_SIZE = 32

THRESHOLD_COMPARE = 0.72
THRESHOLD_EQUIPMENT = 0.5

# ========== ГЛОБАЛЬНЫЕ НАСТРОЙКИ ==========

class GlobalConfig:
    ocr_strategy = "auto"
    ocr_engine = "paddle"
    device_mode = "auto"
    gpu_layers = DEFAULT_N_GPU_LAYERS

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

# ========== УПРАВЛЕНИЕ СЕРВЕРОМ (Server Launcher) ==========

class ServerLauncher:
    _process = None
    _log_file = None

    @classmethod
    def start_server(cls):
        """Запускает llama-server в отдельном процессе с логированием"""
        if cls._process is not None:
            return

        print(f"🚀 Запуск Qwen-VL Server на порту {VL_SERVER_PORT}...")
        
        # Исправлено: используем -ngl вместо --ngl для совместимости
        cmd = [
            "llama-server",
            "--model", PATH_TO_VL_MODEL,
            "--mmproj", PATH_TO_VL_MMPROJ,
            "--host", VL_SERVER_HOST,
            "--port", VL_SERVER_PORT,
            "--ctx-size", "8192",
            "--n-predict", "2048",
            "--temp", "0.1",
            "--n-gpu-layers", "99"  # <--- ИСПРАВЛЕНИЕ: один дефис
        ]

        try:
            # Открываем файл для логов вместо DEVNULL
            cls._log_file = open("llama_server.log", "w", encoding="utf-8")
            
            cls._process = subprocess.Popen(
                cmd, 
                stdout=cls._log_file, 
                stderr=cls._log_file
            )
            
            print("⏳ Ожидание готовности сервера (до 60 сек)...")
            
            # Цикл ожидания с проверкой жизни процесса
            for i in range(60):
                # 1. Проверяем, не умер ли процесс
                return_code = cls._process.poll()
                if return_code is not None:
                    cls.print_error_log()
                    raise RuntimeError(f"❌ Server process died immediately with code {return_code}. See llama_server.log")

                # 2. Проверяем здоровье сервера
                try:
                    requests.get(f"http://{VL_SERVER_HOST}:{VL_SERVER_PORT}/health", timeout=1)
                    print("✅ Сервер Qwen-VL готов к работе!")
                    return
                except requests.exceptions.ConnectionError:
                    time.sleep(1)
            
            # Если тайм-аут
            cls.print_error_log()
            cls.stop_server()
            raise TimeoutError("Не удалось подключиться к llama-server за 60 секунд. Проверьте llama_server.log")

        except FileNotFoundError:
            print("❌ Ошибка: команда 'llama-server' не найдена в PATH.")
            raise

    @classmethod
    def print_error_log(cls):
        """Выводит последние строки лога при ошибке"""
        print("\n" + "!"*50)
        print("ПОСЛЕДНИЕ СТРОКИ ИЗ llama_server.log:")
        try:
            with open("llama_server.log", "r", encoding="utf-8") as f:
                lines = f.readlines()
                for line in lines[-15:]:
                    print("   " + line.strip())
        except:
            print("   (Не удалось прочитать лог)")
        print("!"*50 + "\n")

    @classmethod
    def stop_server(cls):
        """Останавливает сервер"""
        if cls._process:
            print("🛑 Остановка Qwen-VL Server...")
            cls._process.terminate()
            try:
                cls._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                cls._process.kill()
            cls._process = None
        
        if cls._log_file:
            cls._log_file.close()
            cls._log_file = None
            
        print("✅ Сервер остановлен. VRAM свободна.")
        time.sleep(2)

# ========== МЕНЕДЖЕР МОДЕЛЕЙ ==========

class ModelManager:
    _llm = None
    _embedders = {}
    _ocr_paddle = None

    @classmethod
    def get_llm(cls):
        """Загружает только текстовую LLM (Qwen-14B)"""
        if cls._llm is None:
            ServerLauncher.stop_server()
            
            Timer.start("Загрузка LLM")
            if not os.path.exists(PATH_TO_LLM):
                raise FileNotFoundError(f"LLM not found: {PATH_TO_LLM}")
            
            n_gpu = 0
            if GlobalConfig.device_mode == "cpu":
                n_gpu = 0
                print("💻 Режим: CPU Only")
            else:
                n_gpu = GlobalConfig.gpu_layers
                if torch.cuda.is_available():
                    device_count = torch.cuda.device_count()
                    print(f"🚀 Режим: GPU ({device_count} устройств, слоев: {n_gpu})")
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
                flash_attn=True
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
        """Инициализирует PaddleOCR"""
        if GlobalConfig.ocr_engine == "paddle":
            if cls._ocr_paddle is None:
                use_gpu = GlobalConfig.device_mode != "cpu" and torch.cuda.is_available()
                import warnings
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    cls._ocr_paddle = PaddleOCR(use_angle_cls=True, lang='ru', show_log=False, use_gpu=use_gpu)
            return cls._ocr_paddle
        return None

    @classmethod
    def unload_all(cls):
        print("🧹 Освобождение ресурсов...")
        ServerLauncher.stop_server()
        
        if cls._llm:
            try: cls._llm.close()
            except: pass
            cls._llm = None
        
        cls._embedders = {}
        cls._ocr_paddle = None
        
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

# ========== ЗАГРУЗЧИК ДОКУМЕНТОВ ==========

class ExtractionLog:
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
    def _read_pdf_smart(path: str) -> List[str]:
        pages_text = []
        try:
            doc = pymupdf.open(path)
            if doc.is_encrypted:
                print(f"❌ PDF защищён: {path}")
                return []
            
            total_pages = len(doc)
            
            force_vl_server = (GlobalConfig.ocr_strategy == "force" and GlobalConfig.ocr_engine == "qwen-vl-server")
            
            if force_vl_server:
                print(f"   👁️ Режим Qwen-VL Server (Force): Обработка {total_pages} страниц...")
                for i in range(total_pages):
                    print(f"   📡 Отправка стр. {i+1}/{total_pages} на сервер...", end="\r")
                    
                    page = doc[i]
                    pix = page.get_pixmap(matrix=pymupdf.Matrix(2.0, 2.0))
                    img_data = pix.tobytes("png")
                    
                    text = DocumentLoader._send_image_bytes_to_server(img_data)
                    
                    if not text or "Error" in text:
                        text = f"[Сбой чтения страницы {i+1} через сервер]"
                        
                    pages_text.append(text)
                
                print("\n   ✅ Обработка сервером завершена.")
                doc.close()
                return pages_text

            with pdfplumber.open(path) as pdf:
                for i, page in enumerate(pdf.pages):
                    print(f"   📄 Обработка страницы {i+1}/{total_pages}", end="\r")
                    
                    page_content = ""
                    tables = page.extract_tables()
                    if tables:
                        for table in tables:
                            page_content += DocumentLoader._preprocess_table_to_string(table) + "\n\n"
                    
                    text = page.extract_text()
                    if text:
                        page_content += text + "\n"
                    
                    needs_ocr = False
                    if GlobalConfig.ocr_strategy == "force":
                        needs_ocr = True 
                    elif GlobalConfig.ocr_strategy == "auto":
                        if len(page_content.strip()) < 50:
                            needs_ocr = True
                    
                    if needs_ocr and GlobalConfig.ocr_engine != "skip" and GlobalConfig.ocr_engine != "qwen-vl-server":
                        ocr_text = DocumentLoader._ocr_page_via_pymupdf(path, i)
                        page_content += "\n[OCR]:\n" + ocr_text

                    if page_content.strip():
                        pages_text.append(page_content)
            
            print()
            doc.close()
            
        except Exception as e:
            print(f"❌ Ошибка чтения PDF: {e}")
            return []
            
        return pages_text

    @staticmethod
    def _preprocess_table_to_string(table: List[List[Any]]) -> str:
        if not table: return ""

        cleaned_table = []
        for row in table:
            cleaned_row = [str(cell).strip().replace('\n', ' ') if cell is not None else "" for cell in row]
            if any(cleaned_row):
                cleaned_table.append(cleaned_row)
        
        if not cleaned_table: return ""

        text_lines = []
        current_item_number = None
        
        for row in cleaned_table:
            first_col = row[0] if row else ""
            is_new_item = any([
                first_col.isdigit() and len(first_col) <= 3,
                first_col.lower() == "№",
                "наименование" in "".join(row).lower() and "характеристик" in "".join(row).lower()
            ])
            
            if is_new_item or (first_col.isdigit() and len(first_col) <= 3):
                if current_item_number and current_item_number != first_col:
                    text_lines.append("---ITEM_SEPARATOR---")
                current_item_number = first_col
                text_lines.append("[ITEM] " + " | ".join([c for c in row if c]))
            elif len(first_col) == 0 and len(row) > 1 and row[1]: 
                text_lines.append("[SPEC] " + " | ".join([c for c in row if c]))
            else:
                text_lines.append(" | ".join([cell for cell in row if cell]))
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
        return ""

    @staticmethod
    def _send_image_bytes_to_server(image_bytes: bytes) -> str:
        b64_img = base64.b64encode(image_bytes).decode('utf-8')
        sys_prompt = "Ты — эксперт по OCR. Твоя задача — переписать ВЕСЬ текст с изображения. Сохраняй структуру таблиц (используй Markdown). Если видишь списки характеристик, пиши их построчно."
        
        payload = {
            "messages": [
                {"role": "system", "content": sys_prompt},
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Прочитай этот документ и выведи текст."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64_img}"}}
                    ]
                }
            ],
            "max_tokens": 4096, 
            "temperature": 0.1,
            "stream": False
        }
        
        try:
            response = requests.post(VL_SERVER_URL, json=payload, timeout=120)
            if response.status_code == 200:
                res_json = response.json()
                return res_json['choices'][0]['message']['content']
            else:
                return f"[Server Error: {response.status_code}]"
        except Exception as e:
            return f"[Connection Failed: {e}]"

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
    try:
        text = re.sub(r'```json|```', '', text).strip()
        match = re.search(r'(\{.*\}|\[.*\])', text, re.DOTALL)
        if match: return json.loads(match.group())
    except: pass
    return None

def truncate_text(text: str, max_tokens: int = MAX_PROMPT_TOKENS) -> str:
    max_chars = max_tokens * 3
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

# ========== ВЕТКА EQUIPMENT ==========

def _group_fragmented_requirements(reqs: List[Dict]) -> List[Dict]:
    """Группирует разбитые характеристики одного товара (из v6)"""
    if not reqs: return []
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
            item_name.lower().startswith(("количество", "тип", "объем", "частота", "форм-фактор", "поддержка", "интерфейс", "наличие", "максимальное")),
            len(item_name) < 50 and not is_main_item,
            "не менее" in specs or "не более" in specs,
            item_name.lower().startswith("кол-во")
        ])
        
        if is_main_item or (not is_sub_spec and not current_item):
            if current_item: 
                grouped.append(current_item)
                if merged_count > 0: print(f"      ✓ Сгруппировано: '{current_item['item'][:50]}...' (+{merged_count})")
                merged_count = 0
            current_item = {"item": item_name, "specs": specs}
        elif is_sub_spec and current_item:
            sep = "; " if current_item["specs"] else ""
            current_item["specs"] += f"{sep}{item_name}: {specs}"
            merged_count += 1
        else:
            if current_item: 
                grouped.append(current_item)
                merged_count = 0
            current_item = req
    
    if current_item: grouped.append(current_item)
    return grouped

def eq_extract_node(state):
    Timer.start("eq_extract")
    print("📊 [1/4] Извлечение...")
    
    ExtractionLog.logs = []
    
    # 1. Читаем файлы
    if GlobalConfig.ocr_engine == "qwen-vl-server" and GlobalConfig.ocr_strategy == "force":
        try:
            ServerLauncher.start_server()
            pages_spec = DocumentLoader.read_file(state["input_1"])
            pages_offer = DocumentLoader.read_file(state["input_2"])
        finally:
            ServerLauncher.stop_server()
    else:
        pages_spec = DocumentLoader.read_file(state["input_1"])
        pages_offer = DocumentLoader.read_file(state["input_2"])
    
    llm = ModelManager.get_llm()
    all_reqs = []
    all_offers = []
    
    # 2. Анализ ТЗ (Промпт восстановлен из v6)
    print(f"   🧠 Анализ ТЗ ({len(pages_spec)} стр)...")
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
[SPEC] Тип корпуса – Rack 19"
```
-> {{"item": "Сервер DELL PowerEdge R450", "specs": "Тип корпуса: Rack 19..."}}

Формат ответа строго JSON: [{{"item": "...", "specs": "..."}}]<|im_end|>
<|im_start|>user
{safe_page}<|im_end|>
<|im_start|>assistant
"""
        out = llm(prompt, max_tokens=2048, temperature=0.1)
        res = parse_json_garbage(out['choices'][0]['text'])
        if isinstance(res, list): all_reqs.extend(res)

    # 3. Анализ Сметы (Промпт восстановлен из v6)
    print(f"   🧠 Анализ Сметы ({len(pages_offer)} стр)...")
    for i, page in enumerate(pages_offer):
        safe_page = truncate_text(page)
        prompt = f"""<|im_start|>system
Извлеки товары из сметы/коммерческого предложения.

Формат: [{{"name": "Полное название товара", "price": "Цена с единицей", "specs": "Все характеристики"}}]

ВАЖНО: Один товар = один объект. Если указана цена/артикул - это отдельная позиция.<|im_end|>
<|im_start|>user
{safe_page}<|im_end|>
<|im_start|>assistant
"""
        out = llm(prompt, max_tokens=2048, temperature=0.1)
        res = parse_json_garbage(out['choices'][0]['text'])
        if isinstance(res, list): all_offers.extend(res)

    # 4. Пост-обработка
    print("   🔧 Группировка характеристик...")
    all_reqs = _group_fragmented_requirements(all_reqs)
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
    req_txt = [f"{r.get('item','')} {r.get('specs','')}" for r in reqs]
    off_txt = [f"{o.get('name','')} {o.get('specs','')}" for o in offers]
    
    emb_req = embedder.encode(req_txt, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=False)
    emb_off = embedder.encode(off_txt, batch_size=EMBEDDING_BATCH_SIZE, show_progress_bar=False)
    
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
JSON: {{"pass": true/false, "reason": "почему"}}<|im_end|>
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
    report = f"# 🛠 Анализ закупки\n**Дата:** {time.strftime('%Y-%m-%d %H:%M')}\n**Соответствие:** {ok}/{total}\n\n"
    for r in res:
        icon = "✅" if r['status'] == "OK" else ("❓" if r['status'] == "MISSING" else "❌")
        item_name = r.get('item', 'Неизвестно')[:80]
        report += f"### {icon} {item_name}\n"
        if r['status'] != "MISSING":
            offer_name = r.get('offer', 'Не указано')
            specs = r.get('specs', 'Не указаны')[:500]
            report += f"**Товар:** {offer_name} ({r.get('price')})\n**Спецификация:** {specs}\n"
        report += f"**Вывод:** {r.get('reason')}\n\n---\n"
    report += ExtractionLog.summary()
    Timer.stop("eq_report")
    return {"final_report": report}

def build_graph(task_type: str):
    workflow = StateGraph(UnifiedState)
    if task_type == "compare":
        # Заглушка для compare ветки
        return None 
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

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Unified AI Agent v7 (Fixed & Restored v6 Logic)")
    parser.add_argument("--task", type=str, choices=["equipment"], required=True)
    parser.add_argument("file1", help="ТЗ")
    parser.add_argument("file2", help="Смета")
    parser.add_argument("--ocr-strategy", choices=["auto", "force", "skip"], default="force")
    parser.add_argument("--ocr-engine", choices=["paddle", "qwen-vl-server"], default="qwen-vl-server")
    parser.add_argument("--device", choices=["auto", "cpu", "gpu"], default="auto")
    
    args = parser.parse_args()
    if not os.path.exists(args.file1): exit(f"❌ {args.file1}")
    if not os.path.exists(args.file2): exit(f"❌ {args.file2}")
    
    GlobalConfig.ocr_strategy = args.ocr_strategy
    GlobalConfig.ocr_engine = args.ocr_engine
    GlobalConfig.device_mode = args.device
    
    def cleanup(signum, frame):
        ServerLauncher.stop_server()
        exit(1)
    signal.signal(signal.SIGINT, cleanup)
    
    print("="*50)
    print(f"🚀 ЗАДАЧА: {args.task.upper()}")
    print(f"📄 1: {os.path.basename(args.file1)}")
    print(f"📄 2: {os.path.basename(args.file2)}")
    print(f"🔧 OCR Engine: {args.ocr_engine} (Strategy: {args.ocr_strategy})")
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
        ServerLauncher.stop_server()
        ModelManager.unload_all()