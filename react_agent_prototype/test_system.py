"""
Скрипт для тестирования полной системы.

Этот скрипт:
1. Запускает MCP-серверы в фоне
2. Дает им время на инициализацию
3. Запускает ReAct-агент
4. Собирает результаты и логи
5. Останавливает все процессы
"""

import subprocess
import time
import requests
import sys
import os
import signal
from typing import Optional

# Цвета для вывода
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

class SystemTester:
    def __init__(self):
        self.processes = []
        self.project_dir = os.path.dirname(os.path.abspath(__file__))
        self.venv_dir = os.path.join(self.project_dir, "venv")
    
    def log(self, message: str, level: str = "INFO"):
        """Выводит логированное сообщение."""
        colors = {
            "INFO": BLUE,
            "SUCCESS": GREEN,
            "WARNING": YELLOW,
            "ERROR": RED
        }
        color = colors.get(level, BLUE)
        print(f"{color}[{level}]{RESET} {message}")
    
    def run_command(self, cmd: list, name: str, wait: bool = False) -> Optional[subprocess.Popen]:
        """Запускает команду в отдельном процессе."""
        self.log(f"Starting {name}...", "INFO")
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd=self.project_dir
            )
            
            if wait:
                process.wait()
            else:
                self.processes.append((name, process))
            
            return process
        except Exception as e:
            self.log(f"Failed to start {name}: {e}", "ERROR")
            return None
    
    def wait_for_server(self, url: str, timeout: int = 30) -> bool:
        """Ожидает, пока сервер станет доступным."""
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(f"{url}/health", timeout=2)
                if response.status_code == 200:
                    self.log(f"Server {url} is ready", "SUCCESS")
                    return True
            except requests.exceptions.RequestException:
                pass
            
            time.sleep(1)
        
        self.log(f"Server {url} did not respond within {timeout}s", "WARNING")
        return False
    
    def start_servers(self):
        """Запускает MCP-серверы."""
        self.log("Starting MCP Servers...", "INFO")
        
        # Document Server
        doc_cmd = [
            sys.executable, "-m", "uvicorn",
            "mcp_document_server:app",
            "--host", "0.0.0.0",
            "--port", "8001",
            "--log-level", "info"
        ]
        self.run_command(doc_cmd, "Document Server")
        time.sleep(2)
        
        # Legal Server
        legal_cmd = [
            sys.executable, "-m", "uvicorn",
            "mcp_legal_server:app",
            "--host", "0.0.0.0",
            "--port", "8002",
            "--log-level", "info"
        ]
        self.run_command(legal_cmd, "Legal Server")
        time.sleep(2)
        
        # Проверяем доступность серверов
        self.log("Waiting for servers to be ready...", "INFO")
        
        doc_ready = self.wait_for_server("http://localhost:8001")
        legal_ready = self.wait_for_server("http://localhost:8002")
        
        if not (doc_ready and legal_ready):
            self.log("Not all servers are ready. Continuing anyway...", "WARNING")
        
        return doc_ready and legal_ready
    
    def run_agent(self):
        """Запускает ReAct-агент."""
        self.log("Running ReAct Agent...", "INFO")
        
        agent_cmd = [sys.executable, "react_agent_http.py"]
        process = self.run_command(agent_cmd, "ReAct Agent", wait=True)
        
        return process
    
    def cleanup(self):
        """Останавливает все процессы."""
        self.log("Cleaning up processes...", "INFO")
        
        for name, process in self.processes:
            try:
                self.log(f"Stopping {name}...", "INFO")
                process.terminate()
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.log(f"Force killing {name}...", "WARNING")
                process.kill()
            except Exception as e:
                self.log(f"Error stopping {name}: {e}", "ERROR")
    
    def run(self):
        """Запускает полный тест системы."""
        try:
            self.log("=== ReAct Agent System Test ===", "INFO")
            
            # Запускаем серверы
            if not self.start_servers():
                self.log("Servers failed to start properly", "WARNING")
            
            time.sleep(2)
            
            # Запускаем агент
            self.run_agent()
            
            self.log("=== Test Completed ===", "SUCCESS")
        
        except KeyboardInterrupt:
            self.log("Test interrupted by user", "WARNING")
        
        except Exception as e:
            self.log(f"Test failed with error: {e}", "ERROR")
        
        finally:
            self.cleanup()

if __name__ == "__main__":
    tester = SystemTester()
    tester.run()
