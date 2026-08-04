import sys
from pathlib import Path

# Ensure ROOT_DIR is in sys.path when running inside Vercel serverless environment
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from traffic_ai.api.main import app as raw_app  # noqa: E402


async def app(scope, receive, send):
    if scope.get("type") == "http":
        headers = dict(scope.get("headers", []))
        matched_path = headers.get(b"x-matched-path", b"").decode("utf-8")
        forwarded_uri = headers.get(b"x-forwarded-uri", b"").decode("utf-8")
        raw_url = headers.get(b"x-url", b"").decode("utf-8")

        current_path = scope.get("path", "")

        # When Vercel rewrites /demo/analyze -> /api/index.py, scope['path'] becomes /api/index.py or /api/index
        # We restore the real path from Vercel edge headers (x-matched-path or x-forwarded-uri)
        target_path = matched_path or forwarded_uri or raw_url
        if target_path:
            clean_path = target_path.split("?")[0]
            if clean_path and clean_path != current_path and not clean_path.endswith("index.py"):
                scope["path"] = clean_path

    await raw_app(scope, receive, send)


__all__ = ["app"]
