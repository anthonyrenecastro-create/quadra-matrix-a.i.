#!/bin/bash
# Quadra Matrix Neural Command Center - Quick Start

echo "⚡ Quadra Matrix Neural Command Center Setup"
echo "=============================================="
echo ""

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker not found. Please install Docker first."
    exit 1
fi

if ! command -v docker-compose &> /dev/null; then
    echo "❌ Docker Compose not found. Please install Docker Compose first."
    exit 1
fi

echo "✅ Docker and Docker Compose found"
echo ""

# Build and start services
echo "🏗️  Building containers..."
cd dashboard
docker-compose build

echo ""
echo "🚀 Starting services..."
docker-compose up -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 5

# Check backend health
echo "🔍 Checking backend..."
curl -s http://localhost:8000/ > /dev/null
if [ $? -eq 0 ]; then
    echo "✅ Backend is running"
else
    echo "⚠️  Backend may still be starting..."
fi

echo ""
echo "═══════════════════════════════════════════"
echo "✨ Neural Command Center is ready!"
echo "═══════════════════════════════════════════"
echo ""
echo "📍 Access the dashboard:"
echo "   Frontend: http://localhost:3000"
echo "   Backend API: http://localhost:8000"
echo "   API Docs: http://localhost:8000/docs"
echo ""
echo "🎮 Quick Commands:"
echo "   View logs:    docker-compose -f dashboard/docker-compose.yml logs -f"
echo "   Stop:         docker-compose -f dashboard/docker-compose.yml down"
echo "   Restart:      docker-compose -f dashboard/docker-compose.yml restart"
echo ""
echo "📚 Documentation: dashboard/README.md"
echo ""
