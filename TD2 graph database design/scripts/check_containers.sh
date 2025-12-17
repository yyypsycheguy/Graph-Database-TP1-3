#!/bin/bash

# Colors for output
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m' # No Color

echo "Checking Docker containers..."

# List of required services
services=("postgres_db" "neo4j_db" "fastapi_app")

all_running=true

for service in "${services[@]}"; do
    status=$(docker inspect -f '{{.State.Running}}' $service 2>/dev/null)

    if [ "$status" == "true" ]; then
        echo -e "${GREEN}[OK] $service is running${NC}"
    else
        echo -e "${RED}[ERROR] $service is NOT running${NC}"
        all_running=false
    fi
done

if [ "$all_running" = true ]; then
    echo -e "${GREEN}All containers are running!${NC}"
else
    echo -e "${RED}Some containers are not running. Please start them with docker compose up -d${NC}"
    exit 1
fi

# Optional: quick health checks
echo "Checking Postgres connectivity..."
docker exec -i postgres_db pg_isready -U postgres -d shop

echo "Checking Neo4j connectivity..."
docker exec -i neo4j_db cypher-shell -u neo4j -p your_password "RETURN 1" 2>/dev/null && echo "Neo4j is ready" || echo "Neo4j not ready"
