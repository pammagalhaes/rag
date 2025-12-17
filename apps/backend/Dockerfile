# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app


RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*


COPY apps/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

COPY apps/backend/src /app/src
COPY apps/backend/configs /app/configs


COPY packages/rag-core/src/rag_core /app/rag_core


ENV PYTHONPATH="/app/src:/app/configs:/app:${PYTHONPATH}"

EXPOSE 8000


CMD ["uvicorn", "rag_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
