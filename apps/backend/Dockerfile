# backend/Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Dependências do sistema
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    tesseract-ocr \
    poppler-utils \
    libgl1 \
 && rm -rf /var/lib/apt/lists/*

# Instala dependências Python
COPY apps/backend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# Copia código do backend
COPY apps/backend/src /app/src
COPY apps/backend/configs /app/configs

# Copia código do rag-co
COPY packages/rag-core/src/rag_core /app/rag_core

# Ajusta PYTHONPATH para Python enxergar backend, configs e rag_core
ENV PYTHONPATH="/app/src:/app/configs:/app/rag_core:${PYTHONPATH}"

EXPOSE 8000

# Comando para rodar o FastAPI
CMD ["uvicorn", "rag_backend.main:app", "--host", "0.0.0.0", "--port", "8000"]
