# ./app/Dockerfile
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your FastAPI app and ETL code
COPY . .

# Make sure your start script is executable
RUN chmod +x /app/start.sh

# Expose FastAPI port
EXPOSE 8000

# Default command (calls your start script)
CMD ["bash", "start.sh"]
