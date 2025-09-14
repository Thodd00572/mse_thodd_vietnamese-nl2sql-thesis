# Multi-stage Docker build for Vietnamese NL2SQL Thesis Application
FROM node:18-alpine AS frontend-builder

# Set working directory for frontend
WORKDIR /app/frontend

# Copy frontend package files
COPY code/frontend/package*.json ./
RUN npm ci --only=production

# Copy frontend source
COPY code/frontend/ ./
RUN npm run build

# Python backend stage
FROM python:3.11-slim AS backend

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV SKIP_LOCAL_MODELS=true
ENV PORT=8000

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy backend requirements
COPY code/backend/requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend source code
COPY code/backend/ ./backend/
COPY data/ ./data/

# Copy frontend build from previous stage
COPY --from=frontend-builder /app/frontend/out ./frontend/out/
COPY --from=frontend-builder /app/frontend/.next ./frontend/.next/
COPY --from=frontend-builder /app/frontend/package.json ./frontend/

# Create non-root user
RUN useradd --create-home --shell /bin/bash app
RUN chown -R app:app /app
USER app

# Expose port
EXPOSE $PORT

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

# Start command
CMD ["python", "-m", "uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
