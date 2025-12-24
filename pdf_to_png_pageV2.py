import pymupdf
import base64
import json
import subprocess
import sys
import os
import tempfile
import time
import socket
import signal
import urllib.request
import ssl
from contextlib import contextmanager

# --- Настройки ---
PDF_PATH = "/home/seral/diploma/dev_1_conda/documents/f5jsglrkyfe8p3ogd02g405d9q3r9lfn.pdf"
DPI = 200
SERVER_HOST = "127.0.0.1"
SERVER_PORT = 8081
CURL_ENDPOINT = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/chat/completions"
HEALTH_CHECK_URL = f"http://{SERVER_HOST}:{SERVER_PORT}/v1/models"

# Пути к модели и мультимодальной проекции
MODEL_PATH = "./models/gguf/Qwen3-VL-8B-Q4/Qwen3-VL-8B-Instruct-Q4_K_M.gguf"
MMPROJ_PATH = "./models/gguf/Qwen3-VL-8B-Q4/mmproj-Qwen3-VL-8B-Instruct-F16.gguf"

# Параметры сервера
SERVER_ARGS = [
    "llama-server",
    "--model", MODEL_PATH,
    "--mmproj", MMPROJ_PATH,
    "--host", SERVER_HOST,
    "--port", str(SERVER_PORT),
    "--tensor-split", "50,50",
    "--n-gpu-layers", "-1",
    "--ctx-size", "8192",
    "--batch-size", "256",
    "--jinja",
    "--verbose"
]
# ---------------

@contextmanager
def managed_llama_server():
    """
    Контекстный менеджер для запуска и остановки llama-server.
    """
    server_process = None
    try:
        print("🚀 Запускаем llama-server...")
        server_process = subprocess.Popen(
            SERVER_ARGS,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            bufsize=1,
            universal_newlines=True
        )

        import threading
        def log_output(pipe, prefix):
            """
            Читает вывод из pipe (stdout/stderr) и печатает его с префиксом.
            Обрабатывает возможные ошибки декодирования.
            """
            try:
                # Используем decode с обработкой ошибок
                for line_bytes in pipe:
                    # Декодируем байты в строку, заменяя проблемные символы
                    line = line_bytes.decode('utf-8', errors='replace').strip()
                    if "load_tensors: offloaded" in line:
                        print(f"[llama-server {prefix}]: 🟢 {line}")
                    elif "srv  log_server_r: request: POST" in line:
                        print(f"[llama-server {prefix}]: ⚠️ {line}")
                    else:
                        print(f"[llama-server {prefix}]: {line}")
            except UnicodeDecodeError as e:
                print(f"[llama-server {prefix}]: Ошибка декодирования: {e}")
            except Exception as e:
                print(f"[llama-server {prefix}]: Неожиданная ошибка в логировании: {e}")

        threading.Thread(target=log_output, args=(server_process.stdout, "STDOUT"), daemon=True).start()
        threading.Thread(target=log_output, args=(server_process.stderr, "STDERR"), daemon=True).start()

        print(f"⏳ Ожидание готовности сервера ({SERVER_HOST}:{SERVER_PORT})...")
        if not wait_for_server_ready(SERVER_HOST, SERVER_PORT, timeout=600):
            raise RuntimeError("Сервер не готов к работе за отведённое время!")

        print("✅ Сервер полностью готов к обработке запросов!")
        yield server_process

    finally:
        if server_process and server_process.poll() is None:
            print("\n🛑 Завершаем работу llama-server...")
            server_process.terminate()
            try:
                server_process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                print("⚠️ Принудительное завершение сервера (SIGKILL)")
                server_process.kill()
            print("✅ Сервер остановлен")


def wait_for_server_ready(host, port, timeout=600):
    """Проверяет готовность сервера: открытие порта + готовность API"""
    start_time = time.time()
    
    # Этап 1: Проверка открытия порта
    print("🔍 Этап 1/3: Проверка открытия порта...")
    while time.time() - start_time < timeout:
        try:
            with socket.create_connection((host, port), timeout=1):
                print("✅ Порт открыт")
                break
        except (socket.timeout, ConnectionRefusedError, OSError):
            time.sleep(1)
    else:
        print("❌ Таймаут ожидания открытия порта")
        return False

    # Этап 2: Проверка доступности /v1/models (показывает, что API запущено)
    print("🔍 Этап 2/3: Проверка готовности API (/v1/models)...")
    ctx = ssl.create_default_context()
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE

    while time.time() - start_time < timeout:
        try:
            req = urllib.request.Request(HEALTH_CHECK_URL, headers={"Content-Type": "application/json"})
            with urllib.request.urlopen(req, context=ctx, timeout=5) as response:
                if response.status == 200:
                    data = json.loads(response.read().decode())
                    if data.get("data") and len(data["data"]) > 0:
                        print("✅ API готов, модель доступна")
                        break
        except:
            pass
        time.sleep(2)
    else:
        print("❌ Таймаут ожидания готовности API")
        return False

    # Этап 3: Проверка готовности /v1/chat/completions через тестовый запрос
    print("🔍 Этап 3/3: Проверка готовности обработки чат-запросов...")
    test_payload = {
        "model": "qwen-vl",
        "messages": [{"role": "user", "content": [{"type": "text", "text": "ping"}]}],
        "max_tokens": 1
    }

    while time.time() - start_time < timeout:
        try:
            # Отправляем тестовый запрос
            req = urllib.request.Request(
                CURL_ENDPOINT,
                data=json.dumps(test_payload).encode('utf-8'),
                headers={"Content-Type": "application/json"}
            )
            with urllib.request.urlopen(req, context=ctx, timeout=10) as response:
                response_data = json.loads(response.read().decode())
                # Если запрос прошёл без 503 - сервер готов
                if "error" not in response_data or response_data.get("error", {}).get("code") != 503:
                    print("✅ Сервер готов к обработке чат-запросов!")
                    return True
        except urllib.error.HTTPError as e:
            if e.code == 503:
                print("🔄 Модель ещё загружается (HTTP 503)...")
            else:
                print(f"⚠️ Ошибка тестового запроса: {e}")
        except Exception as e:
            print(f"⚠️ Ошибка подключения: {e}")
        
        time.sleep(5)
    
    print("❌ Таймаут ожидания готовности обработки чат-запросов")
    return False


def process_page_to_base64_and_send(pdf_document, page_number):
    """
    Обрабатывает одну страницу PDF: конвертирует в PNG -> base64 -> отправляет запрос -> возвращает ответ модели.
    """
    print(f"📄 Обработка страницы {page_number + 1}/{len(pdf_document)}...")
    page = pdf_document[page_number]

    # Конвертировать страницу в PNG
    pix = page.get_pixmap(dpi=DPI)
    img_data = pix.tobytes("png")

    # Закодировать в base64
    b64_data = base64.b64encode(img_data).decode('utf-8')

    # --- Формирование JSON запроса ---
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "Извлеки весь текст с изображения. Сохрани структуру таблиц используя | для разделения колонок. Не добавляй ничего от себя."
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/png;base64,{b64_data}"
                    }
                }
            ]
        }
    ]

    payload = {
        "model": "qwen-vl",
        "messages": messages,
        "max_tokens": 8192,
        "temperature": 0.1
    }

    # --- Запись payload в временный файл ---
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json', encoding='utf-8') as temp_json:
        json.dump(payload, temp_json, ensure_ascii=False)
        temp_json_path = temp_json.name

    # --- Отправка запроса через curl с использованием файла ---
    try:
        cmd = ["curl", "-s", "-X", "POST", CURL_ENDPOINT,
               "-H", "Content-Type: application/json",
               "-d", f"@{temp_json_path}"]
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)

        # Удаляем временный файл после использования
        os.unlink(temp_json_path)

        # Разбираем JSON ответа
        response_json = json.loads(result.stdout)
        choices = response_json.get("choices", [])
        if not choices:
            print(f"  ⚠️ Предупреждение: Нет вариантов ответа для страницы {page_number + 1}")
            return ""

        message_content = choices[0].get("message", {}).get("content", "")
        if not message_content:
            print(f"  ⚠️ Предупреждение: Нет содержимого сообщения для страницы {page_number + 1}")
            return ""

        print(f"  ✅ Страница {page_number + 1}: Успешно обработана!")
        return message_content

    except subprocess.CalledProcessError as e:
        stderr_output = e.stderr.strip()
        if "503" in stderr_output or "Loading model" in stderr_output:
            print(f"  ⚠️ Сервер ещё не готов (HTTP 503). Повторная попытка через 2 сек...")
            time.sleep(2)
            # Повторяем запрос один раз
            return process_page_to_base64_and_send(pdf_document, page_number)
        print(f"  ❌ Ошибка curl для страницы {page_number + 1}: {stderr_output}")
        if os.path.exists(temp_json_path):
            os.unlink(temp_json_path)
        return f"Ошибка обработки страницы {page_number + 1}"
    except json.JSONDecodeError as e:
        print(f"  ❌ Ошибка разбора JSON для страницы {page_number + 1}: {e}")
        print(f"  📄 Ответ сервера:\n{result.stdout}")
        if os.path.exists(temp_json_path):
            os.unlink(temp_json_path)
        return f"Ошибка разбора ответа страницы {page_number + 1}"
    except Exception as e:
        print(f"  ❌ Неожиданная ошибка для страницы {page_number + 1}: {e}")
        if os.path.exists(temp_json_path):
            os.unlink(temp_json_path)
        return f"Критическая ошибка на странице {page_number + 1}"

def main():
    """Основной процесс: запуск сервера -> обработка PDF -> сохранение результата."""
    # Проверяем существование файлов модели
    for path, name in [(MODEL_PATH, "модель"), (MMPROJ_PATH, "mmproj")]:
        if not os.path.exists(path):
            raise FileNotFoundError(f"❌ Файл {name} не найден по пути: {path}")

    # Запускаем сервер в контекстном менеджере
    with managed_llama_server():
        print(f"\n📂 Открываем PDF файл: {PDF_PATH}")
        try:
            doc = pymupdf.open(PDF_PATH)
        except Exception as e:
            print(f"❌ Ошибка открытия PDF: {e}")
            sys.exit(1)

        total_pages = len(doc)
        print(f"📄 Всего страниц для обработки: {total_pages}\n")

        all_results = []
        for i in range(total_pages):
            page_result = process_page_to_base64_and_send(doc, i)
            all_results.append(page_result)
            time.sleep(0.5)  # Пауза между запросами

        doc.close()
        print("\n✅ Все страницы успешно обработаны!")

        # --- Объединение результатов ---
        print("\n" + "="*60)
        print("📝 ФИНАЛЬНЫЙ РЕЗУЛЬТАТ:")
        print("="*60)
        final_output = "\n\n".join(
            [f"{'='*40} СТРАНИЦА {i+1} {'='*40}\n{content}" for i, content in enumerate(all_results)]
        ) + "\n"

        # --- Сохранение результата в файл ---
        output_dir = "results"
        os.makedirs(output_dir, exist_ok=True)
        output_filename = os.path.join(
            output_dir,
            f"{os.path.splitext(os.path.basename(PDF_PATH))[0]}_extracted_tables.txt"
        )
        
        with open(output_filename, 'w', encoding='utf-8') as f:
            f.write(final_output)
        print(f"\n💾 Результат сохранён в файл: {output_filename}")
        
        print(f"\n📋 Обработано {len(all_results)} страниц.")


if __name__ == "__main__":
    def signal_handler(sig, frame):
        print("\n🛑 Получен сигнал прерывания (Ctrl+C). Завершаем работу корректно...")
        sys.exit(0)
    
    signal.signal(signal.SIGINT, signal_handler)
    
    main()

    