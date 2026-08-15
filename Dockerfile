FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Dependency layer first so source edits don't invalidate the install.
COPY pyproject.toml README.md ./
COPY src ./src

# torch defaults to the CUDA build, which pulls ~3.5 GB of nvidia/ and
# triton/ wheels. This container has no GPU — embeddings run on CPU — so
# that payload is downloaded and stored on every deploy and never executed.
# The CPU index cuts the image from 5.75 GB to ~2 GB.
RUN pip install --upgrade pip \
    && pip install --index-url https://download.pytorch.org/whl/cpu torch \
    && pip install .

COPY . .

# A non-root user: the container needs no write access to its own code, and
# running as root would let a compromise modify the application.
RUN useradd --create-home --uid 1000 edgar && chown -R edgar:edgar /app
USER edgar

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=90s --retries=3 \
    CMD python -c "import urllib.request;urllib.request.urlopen('http://localhost:8000/health')"

# Migrations run before the service accepts traffic; the app itself never
# creates tables, so this is the only thing that defines the schema.
CMD ["sh", "-c", "alembic upgrade head && exec uvicorn edgar_rag.api.main:app --host 0.0.0.0 --port 8000"]
