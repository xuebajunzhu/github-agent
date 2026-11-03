FROM python:3.12-slim

WORKDIR /app

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

COPY requirements.txt requirements-ml.txt ./
ARG INSTALL_ML=true
RUN if [ "$INSTALL_ML" = "true" ]; then \
        pip install --no-cache-dir -r requirements.txt -r requirements-ml.txt; \
    else \
        pip install --no-cache-dir -r requirements.txt; \
    fi

COPY app ./app

# writable state dirs (chroma persistence, file logs) + drop root
RUN useradd --create-home --uid 10001 agent \
    && mkdir -p /app/chroma_data /app/logs \
    && chown -R agent:agent /app
USER agent

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD ["python", "-c", "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)"]

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
