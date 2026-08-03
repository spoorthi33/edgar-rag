"""CLI for Phase 7: run the API service.

python scripts/serve.py
python scripts/serve.py --reload --port 8080
"""

from __future__ import annotations

import argparse
import sys

import uvicorn

from edgar_rag.config import get_settings


def main() -> int:
    settings = get_settings()

    parser = argparse.ArgumentParser(description="Run the edgar-rag API")
    parser.add_argument("--host", default=settings.api_host)
    parser.add_argument("--port", type=int, default=settings.api_port)
    parser.add_argument("--reload", action="store_true", help="restart on code changes")
    args = parser.parse_args()

    print(f"starting on http://{args.host}:{args.port} (docs at /docs)")
    uvicorn.run(
        "edgar_rag.api.main:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
