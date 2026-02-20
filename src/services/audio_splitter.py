import logging
import os
import subprocess
import uuid
from pathlib import Path

logger = logging.getLogger(__name__)

TEMP_DIR = Path("storage/chunks")
TEMP_DIR.mkdir(parents=True, exist_ok=True)


def get_duration(filepath: str) -> float:
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "quiet", "-show_entries", "format=duration",
             "-of", "default=noprint_wrappers=1:nokey=1", filepath],
            capture_output=True, text=True, timeout=10,
        )
        return float(result.stdout.strip())
    except Exception as e:
        logger.warning("Could not get duration: %s", e)
        return 0.0


def split_audio(filepath: str, chunk_seconds: float = 60.0) -> list[str]:
    """Split audio into fixed-length chunks. Returns list of chunk file paths in order."""
    duration = get_duration(filepath)
    if duration <= 0:
        return [filepath]

    batch_id = uuid.uuid4().hex[:8]
    chunks: list[str] = []

    start = 0.0
    idx = 0
    while start < duration:
        chunk_path = str(TEMP_DIR / f"{batch_id}_chunk{idx:03d}.wav")
        cmd = [
            "ffmpeg", "-y", "-v", "quiet",
            "-i", filepath,
            "-ss", str(start),
            "-t", str(chunk_seconds),
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            chunk_path,
        ]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            logger.warning("ffmpeg chunk %d failed: %s", idx, result.stderr)
            break
        chunks.append(chunk_path)
        start += chunk_seconds
        idx += 1

    if not chunks:
        return [filepath]

    logger.info("Split '%s' (%.1fs) into %d chunks of %.0fs each", filepath, duration, len(chunks), chunk_seconds)
    return chunks


def cleanup_chunks(chunk_paths: list[str]):
    for p in chunk_paths:
        try:
            os.remove(p)
        except OSError:
            pass
