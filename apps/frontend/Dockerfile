# frontend/Dockerfile
# apps/frontend/Dockerfile
FROM python:3.11-slim

WORKDIR /app


COPY apps/frontend/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt


COPY apps/frontend /app

EXPOSE 8501

CMD ["streamlit", "run", "src/rag_frontend/app.py", "--server.address=0.0.0.0"]


