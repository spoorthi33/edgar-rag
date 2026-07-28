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

# Replaced with the real app in Phase 7.
CMD ["python", "-c", "import edgar_rag; print(f'edgar-rag {edgar_rag.__version__} ready')"]
