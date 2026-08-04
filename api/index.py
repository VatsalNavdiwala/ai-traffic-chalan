import sys
from pathlib import Path
from urllib.parse import parse_qs

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from traffic_ai.api.main import app as raw_app  # noqa: E402


async def app(scope, receive, send):
    if scope.get("type") == "http":
        query_string = scope.get("query_string", b"").decode("utf-8")
        parsed_qs = parse_qs(query_string)

        if "__path" in parsed_qs and parsed_qs["__path"]:
            scope["path"] = parsed_qs["__path"][0]
        else:
            headers = dict(scope.get("headers", []))
            matched_path = headers.get(b"x-matched-path", b"").decode("utf-8")
            forwarded_uri = headers.get(b"x-forwarded-uri", b"").decode("utf-8")
            target = (matched_path or forwarded_uri).split("?")[0]
            if target and not target.endswith("index.py"):
                scope["path"] = target

    await raw_app(scope, receive, send)


__all__ = ["app"]
