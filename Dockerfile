FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency layer first so source edits don't invalidate the install.
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip && pip install .

COPY . .

EXPOSE 8000

CMD ["uvicorn", "edgar_rag.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
