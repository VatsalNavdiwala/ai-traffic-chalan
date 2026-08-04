import sys
from pathlib import Path

# File path: api/demo/analyze.py
# .parent = api/demo, .parent.parent = api, .parent.parent.parent = ROOT_DIR
ROOT_DIR = Path(__file__).resolve().parent.parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from traffic_ai.api.main import app  # noqa: E402

__all__ = ["app"]
