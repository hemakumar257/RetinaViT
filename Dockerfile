# Production Dockerfile for RetinaViT Clinical API
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
RUN pip install uvicorn fastapi python-multipart

# Copy code and models
COPY src/ ./src/
COPY backend/ ./backend/
COPY checkpoints/deployment/ ./checkpoints/deployment/

# Set PYTHONPATH
ENV PYTHONPATH=/app/src:/app

# Expose API port
EXPOSE 8000

# Run FastAPI
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
