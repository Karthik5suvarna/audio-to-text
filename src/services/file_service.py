import logging
import os
import subprocess
import uuid
from pathlib import Path

import aiofiles
from fastapi import UploadFile

from src.core.config import settings

logger = logging.getLogger(__name__)


class FileService:
    def __init__(self):
        self.upload_dir = Path(settings.upload_dir)
        self.upload_dir.mkdir(parents=True, exist_ok=True)

    async def save_upload(self, file: UploadFile) -> str:
        ext = Path(file.filename).suffix.lower()
        filename = f"{uuid.uuid4().hex}{ext}"
        filepath = self.upload_dir / filename

        async with aiofiles.open(filepath, "wb") as f:
            content = await file.read()
            await f.write(content)

        return str(filepath)

    @staticmethod
    def delete_file(filepath: str):
        try:
            os.remove(filepath)
        except OSError:
            pass

    @staticmethod
    def validate_extension(filename: str) -> bool:
        ext = Path(filename).suffix.lower()
        return ext in settings.allowed_extensions

    @staticmethod
    def get_file_size_mb(filepath: str) -> float:
        return os.path.getsize(filepath) / (1024 * 1024)

    @staticmethod
    def get_audio_duration(filepath: str) -> float:
        """Get audio duration in seconds using ffprobe."""
        try:
            result = subprocess.run(
                [
                    "ffprobe", "-v", "quiet",
                    "-show_entries", "format=duration",
                    "-of", "default=noprint_wrappers=1:nokey=1",
                    filepath,
                ],
                capture_output=True, text=True, timeout=10,
            )
            return round(float(result.stdout.strip()), 2)
        except Exception as e:
            logger.warning("Could not determine audio duration: %s", e)
            return 0.0


file_service = FileService()
