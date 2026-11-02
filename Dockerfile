FROM python:3.14-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    ENVIRONMENT=production \
    MODEL_DIR=/data/models \
    REPORT_DIR=/data/reports

WORKDIR /app

# The versions in requirements.lock are the ones the stored models were trained and verified with.
COPY requirements.lock ./
RUN pip install --no-cache-dir -r requirements.lock

COPY app ./app

# Railway sets PORT. Private networking in projects created after 2025-10-16 is reachable over IPv4.
CMD ["sh", "-c", "exec uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
