from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(ROOT_DIR / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    app_name: str = "AI Traffic Signal"
    app_env: str = "development"
    debug: bool = True
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    database_url: str = "postgresql+asyncpg://traffic:traffic@localhost:5432/traffic_ai"
    database_url_sync: str = "postgresql://traffic:traffic@localhost:5432/traffic_ai"

    redis_url: str = "redis://localhost:6379/0"
    celery_broker_url: str = "amqp://guest:guest@localhost:5672//"
    celery_result_backend: str = "redis://localhost:6379/1"

    yolo_model_path: str = str(ROOT_DIR / "traffic_ai" / "models" / "weights" / "yolo11n.pt")
    yolo_confidence: float = 0.95
    device: str = "cuda"
    ocr_lang: str = "en"

    # radar | stereo | dual_camera | single_camera_demo
    speed_mode: str = "radar"
    speed_limit_kmh: float = 60.0

    vahan_api_url: str = ""
    vahan_api_key: str = ""
    sms_gateway_url: str = ""
    sms_gateway_key: str = ""
    whatsapp_api_url: str = ""
    whatsapp_api_key: str = ""

    evidence_dir: str = str(ROOT_DIR / "traffic_ai" / "logs" / "evidence")


@lru_cache
def get_settings() -> Settings:
    return Settings()
