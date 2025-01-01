#!/bin/bash
echo "🚀 Deploying Mario Analytics Platform..."

# Build and start services
docker-compose up --build -d

echo "✅ Services started!"
echo "📊 Dashboard: http://localhost:5000/dashboard.html"
echo "🔌 API: http://localhost:5000/api/"

# Wait for services to be ready
sleep 10

# Initialize database tables (mock)
echo "📋 Initializing database..."
curl -X POST http://localhost:5000/api/init -H "Content-Type: application/json" -d '{"action": "setup_tables"}' || true

echo "🎮 Mario Analytics Platform is ready!"
