from pydantic_settings import BaseSettings
from typing import List


class Settings(BaseSettings):
    model_id: str = "large-v3"
    device: str = "auto"
    upload_dir: str = "storage/uploads"
    max_file_size_mb: int = 500
    allowed_extensions: list[str] = [
        ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".webm", ".mp4",
    ]
    # Accept devices as a comma-separated string in the env file
    # (e.g. DEVICES=cuda:0,cuda:1) to avoid pydantic attempting JSON decode.
    devices: str = "cuda:0,cuda:1"

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()

# Normalize `devices` into a list[str] for downstream code.
if isinstance(settings.devices, str):
    settings.devices = [d.strip() for d in settings.devices.split(",") if d.strip()]