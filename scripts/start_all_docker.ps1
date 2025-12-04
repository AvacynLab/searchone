# Start full stack with docker-compose (PowerShell)
cd ..
# Build and run containers
docker-compose up -d --build
Write-Host "Containers started. Frontend: http://localhost:2000, Backend: http://localhost:2001, Redis on 2002, SearxNG on 2003"
