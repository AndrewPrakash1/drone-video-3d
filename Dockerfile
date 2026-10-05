# Backend Dockerfile for deployment
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy backend files
COPY backend/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY backend/ ./backend/
COPY data/ ./data/

# Create demo files
RUN python -c "from backend.app.demo_scene import write_demo_files; from pathlib import Path; write_demo_files(Path('data/demo'))"

# Expose port
EXPOSE 8765

# Run the application
CMD ["python", "-m", "uvicorn", "backend.app.main:app", "--host", "0.0.0.0", "--port", "8765"]
