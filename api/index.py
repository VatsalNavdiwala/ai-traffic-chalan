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

        target_path = (matched_path or forwarded_uri or raw_url).split("?")[0]
        if target_path and not target_path.endswith("index.py"):
            scope["path"] = target_path

    await raw_app(scope, receive, send)


__all__ = ["app"]
