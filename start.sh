#!/bin/bash
set -e

echo "==> Starting WC2026 Betting Assistant"

# --- Backend ---
echo ""
echo "[1/2] Starting Python backend..."
cd backend

if [ ! -d ".venv" ]; then
  echo "  Creating virtual environment..."
  python3 -m venv .venv
fi

source .venv/bin/activate

echo "  Installing dependencies..."
pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
  cp .env.example .env
  echo "  Created backend/.env — add your API keys there."
fi

uvicorn main:app --host 0.0.0.0 --port 8000 --reload &
BACKEND_PID=$!
echo "  Backend running at http://localhost:8000 (PID $BACKEND_PID)"

# --- Frontend ---
echo ""
echo "[2/2] Starting Next.js frontend..."
cd ../frontend

npm install --silent

npm run dev &
FRONTEND_PID=$!
echo "  Frontend running at http://localhost:3000 (PID $FRONTEND_PID)"

echo ""
echo "  Dashboard → http://localhost:3000"
echo "  API docs  → http://localhost:8000/docs"
echo ""
echo "  Press Ctrl+C to stop both servers."

# Wait and clean up on exit
trap "kill $BACKEND_PID $FRONTEND_PID 2>/dev/null" EXIT
wait
