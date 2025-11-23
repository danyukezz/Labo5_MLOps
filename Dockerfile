FROM python:3.11-slim

WORKDIR /app

# Copy requirements first for better caching
COPY inference/requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the FastAPI application
COPY inference/main.py .

# Create directory for models and copy the trained model
RUN mkdir -p /models/model
COPY model/ /models/model/

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
  CMD python -c "import requests; requests.get('http://localhost:8000/health')" || exit 1

# Run the application
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
