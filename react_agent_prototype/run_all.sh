#!/bin/bash

# Скрипт для запуска всех компонентов системы на одной машине
# Использует tmux для управления несколькими процессами

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="$PROJECT_DIR/venv"

# Цвета для вывода
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}=== ReAct Agent System Startup ===${NC}"

# Проверяем наличие виртуального окружения
if [ ! -d "$VENV_DIR" ]; then
    echo -e "${YELLOW}Creating virtual environment...${NC}"
    python3.11 -m venv "$VENV_DIR"
fi

# Активируем виртуальное окружение
source "$VENV_DIR/bin/activate"

# Проверяем наличие tmux
if ! command -v tmux &> /dev/null; then
    echo -e "${YELLOW}tmux not found. Installing...${NC}"
    sudo apt-get update && sudo apt-get install -y tmux
fi

# Создаем новую tmux сессию
SESSION_NAME="react-agent-system"

if tmux has-session -t "$SESSION_NAME" 2>/dev/null; then
    echo -e "${YELLOW}Killing existing session...${NC}"
    tmux kill-session -t "$SESSION_NAME"
fi

tmux new-session -d -s "$SESSION_NAME" -x 200 -y 50

# Окно 1: Document Server
echo -e "${GREEN}Starting Document Server on port 8001...${NC}"
tmux new-window -t "$SESSION_NAME" -n "doc-server"
tmux send-keys -t "$SESSION_NAME:doc-server" "cd $PROJECT_DIR && source $VENV_DIR/bin/activate && python3.11 mcp_document_server.py" Enter
sleep 2

# Окно 2: Legal Server
echo -e "${GREEN}Starting Legal Server on port 8002...${NC}"
tmux new-window -t "$SESSION_NAME" -n "legal-server"
tmux send-keys -t "$SESSION_NAME:legal-server" "cd $PROJECT_DIR && source $VENV_DIR/bin/activate && python3.11 mcp_legal_server.py" Enter
sleep 2

# Окно 3: ReAct Agent
echo -e "${GREEN}Starting ReAct Agent...${NC}"
tmux new-window -t "$SESSION_NAME" -n "react-agent"
tmux send-keys -t "$SESSION_NAME:react-agent" "cd $PROJECT_DIR && source $VENV_DIR/bin/activate && python3.11 react_agent_http.py" Enter

# Окно 4: Logs/Monitor
echo -e "${GREEN}Opening monitor window...${NC}"
tmux new-window -t "$SESSION_NAME" -n "monitor"
tmux send-keys -t "$SESSION_NAME:monitor" "cd $PROJECT_DIR && echo 'System is running. Use Ctrl+C to stop.' && sleep infinity" Enter

# Выводим информацию
echo -e "${GREEN}=== System Started ===${NC}"
echo -e "Session name: ${YELLOW}$SESSION_NAME${NC}"
echo -e "Document Server: ${YELLOW}http://localhost:8001${NC}"
echo -e "Legal Server: ${YELLOW}http://localhost:8002${NC}"
echo ""
echo -e "Available tmux windows:"
echo -e "  - ${YELLOW}doc-server${NC}: Document Server"
echo -e "  - ${YELLOW}legal-server${NC}: Legal Server"
echo -e "  - ${YELLOW}react-agent${NC}: ReAct Agent"
echo -e "  - ${YELLOW}monitor${NC}: Monitor/Logs"
echo ""
echo -e "To attach to tmux session: ${YELLOW}tmux attach-session -t $SESSION_NAME${NC}"
echo -e "To kill session: ${YELLOW}tmux kill-session -t $SESSION_NAME${NC}"
echo ""

# Attach к сессии
tmux attach-session -t "$SESSION_NAME"
