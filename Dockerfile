FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/home/plonk/.cache/huggingface \
    TORCH_HOME=/home/plonk/.cache/torch

WORKDIR /app

COPY . .

RUN python -m pip install --upgrade pip \
    && python -m pip install ".[api]" \
    && useradd --create-home --uid 10001 plonk \
    && mkdir -p /home/plonk/.cache/huggingface /home/plonk/.cache/torch \
    && chown -R plonk:plonk /app /home/plonk

USER plonk

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3)" || exit 1

# Keep one worker: every worker would otherwise load its own copy of the models.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
