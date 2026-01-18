#!/bin/bash
# Development Mode - Run without Docker

echo "🔧 Starting in Development Mode"
echo "================================"
echo ""

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python 3 not found"
    exit 1
fi

# Check Node
if ! command -v node &> /dev/null; then
    echo "❌ Node.js not found"
    exit 1
fi

echo "✅ Prerequisites found"
echo ""

# Install backend dependencies
echo "📦 Installing backend dependencies..."
cd dashboard/backend
pip install -r requirements.txt
cd ../..

# Install frontend dependencies
echo "📦 Installing frontend dependencies..."
cd dashboard/frontend
npm install
cd ../..

# Start backend in background
echo "🚀 Starting backend..."
cd dashboard/backend
python3 main.py &
BACKEND_PID=$!
cd ../..

# Wait for backend
sleep 3

# Start frontend
echo "🚀 Starting frontend..."
cd dashboard/frontend
npm run dev &
FRONTEND_PID=$!
cd ../..

echo ""
echo "═══════════════════════════════════════════"
echo "✨ Development servers running!"
echo "═══════════════════════════════════════════"
echo ""
echo "📍 Access:"
echo "   Frontend: http://localhost:3000"
echo "   Backend:  http://localhost:8000"
echo ""
echo "Press Ctrl+C to stop all servers"
echo ""

# Wait for user interrupt
trap "kill $BACKEND_PID $FRONTEND_PID; exit" INT
wait
