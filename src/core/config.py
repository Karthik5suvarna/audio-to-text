from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    model_id: str = "large-v3"
    device: str = "auto"
    upload_dir: str = "storage/uploads"
    max_file_size_mb: int = 500
    allowed_extensions: list[str] = [
        ".wav", ".mp3", ".flac", ".ogg", ".m4a", ".wma", ".aac", ".webm", ".mp4",
    ]
    devices: list[str] = ["cuda:0", "cuda:1"] 

    model_config = {"env_file": ".env", "env_file_encoding": "utf-8"}


settings = Settings()
