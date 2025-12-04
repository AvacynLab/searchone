# Start Redis using Docker (PowerShell)
# Requires Docker Desktop installed
docker run -d --name searchone-redis -p 2002:6379 redis:7

Write-Host "Redis container started on localhost:2002"
