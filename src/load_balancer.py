"""
Chunk-parallel orchestrator:
  1. Receives one audio file
  2. Splits into N-second chunks (default 60s)
  3. Distributes chunks across backends with per-backend concurrency control
  4. Merges transcripts in order
  5. Supports SSE streaming for progressive results
"""
import asyncio
import json
import logging
import os
import time
import uuid
from pathlib import Path

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
from sse_starlette.sse import EventSourceResponse
import httpx

from src.services.audio_splitter import split_audio, cleanup_chunks, get_duration
from src.core.config import settings

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)

BACKEND_URLS_ENV = os.environ.get("BACKEND_URLS", "http://127.0.0.1:8001,http://127.0.0.1:8002,http://127.0.0.1:8003")
BACKENDS = [u.strip() for u in BACKEND_URLS_ENV.split(",") if u.strip()]
CHUNK_SECONDS = float(os.environ.get("CHUNK_SECONDS", "60"))
CONCURRENCY_PER_BACKEND = int(os.environ.get("CONCURRENCY_PER_BACKEND", "1"))
UPLOAD_DIR = Path("storage/uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

if not BACKENDS:
    raise RuntimeError("Set BACKEND_URLS (comma-separated) with at least one Whisper backend URL.")

# Per-backend semaphores: limit concurrent requests to each backend
_backend_semas: dict[str, asyncio.Semaphore] = {}

num_gpus = len(settings.devices)

def _get_semaphore(backend: str) -> asyncio.Semaphore:
    if backend not in _backend_semas:
        _backend_semas[backend] = asyncio.Semaphore(CONCURRENCY_PER_BACKEND)
    return _backend_semas[backend]


app = FastAPI(title="Audio to Text (Chunk-Parallel)", version="0.3.0")


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backends": BACKENDS,
        "chunk_seconds": CHUNK_SECONDS,
        "concurrency_per_backend": CONCURRENCY_PER_BACKEND,
    }


async def _send_chunk(
    client: httpx.AsyncClient,
    backend_url: str,
    chunk_path: str,
    chunk_index: int,
    gpu_id: int  # Add GPU ID to track which GPU should process the chunk
) -> dict:
    """Send one chunk to a backend with concurrency control. Stream file handle directly."""
    url = f"{backend_url.rstrip('/')}/api/v1/transcribe"
    filename = Path(chunk_path).name
    sema = _get_semaphore(backend_url)

    async with sema:
        start = time.perf_counter()
        with open(chunk_path, "rb") as f:
            response = await client.post(
                url,
                files={"file": (filename, f, "audio/wav")},
                json={"gpu_id": gpu_id} 
            )
        elapsed = round(time.perf_counter() - start, 3)

    if response.status_code != 200:
        logger.warning("Backend %s returned %d for chunk %d: %s", backend_url, response.status_code, chunk_index, response.text)
        return {"index": chunk_index, "text": "", "error": response.text, "elapsed": elapsed}

    data = response.json()
    logger.info("Chunk %d done by %s in %.1fs", chunk_index, backend_url, elapsed)
    return {"index": chunk_index, "text": data.get("text", ""), "backend": backend_url, "elapsed": elapsed}


async def _process_file(file: UploadFile) -> tuple[str, list[str], float, float]:
    """Save upload, get duration, split into chunks. Returns (temp_path, chunk_paths, duration, size_mb)."""
    ext = Path(file.filename).suffix.lower()
    temp_path = str(UPLOAD_DIR / f"{uuid.uuid4().hex}{ext}")
    content = await file.read()
    with open(temp_path, "wb") as f:
        f.write(content)

    duration = get_duration(temp_path)
    size_mb = round(len(content) / (1024 * 1024), 2)
    chunk_paths = await asyncio.to_thread(split_audio, temp_path, CHUNK_SECONDS)
    return temp_path, chunk_paths, duration, size_mb


def _fmt(sec: float) -> str:
    return f"{sec / 60:.1f} min" if sec >= 60 else f"{sec:.1f} sec"


# --- Standard endpoint: wait for all chunks, return combined result ---

@app.post("/api/v1/transcribe")
async def transcribe(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    total_start = time.perf_counter()
    temp_path, chunk_paths, audio_duration, file_size_mb = await _process_file(file)
    num_chunks = len(chunk_paths)
    logger.info("Audio: %.1fs, chunks: %d (%.0fs each), backends: %d", audio_duration, num_chunks, CHUNK_SECONDS, len(BACKENDS))

    try:
        async with httpx.AsyncClient(timeout=600.0) as client:
            tasks = [
                _send_chunk(client, BACKENDS[i % len(BACKENDS)], chunk_paths[i], i)
                for i in range(num_chunks)
            ]
            results = await asyncio.gather(*tasks)

        results_sorted = sorted(results, key=lambda r: r["index"])
        combined_text = " ".join(r["text"].strip() for r in results_sorted if r.get("text"))
        total_time = time.perf_counter() - total_start
        speed_factor = round(audio_duration / total_time, 2) if total_time > 0 else 0.0

        return JSONResponse(content={
            "text": combined_text,
            "file_name": file.filename,
            "file_size_mb": file_size_mb,
            "audio_length_display": _fmt(audio_duration),
            "total_processing_time_display": _fmt(total_time),
            "speed_factor": speed_factor,
            "num_chunks": num_chunks,
            "chunk_seconds": CHUNK_SECONDS,
            "backends_used": len(BACKENDS),
        })
    finally:
        try:
            os.remove(temp_path)
        except OSError:
            pass
        cleanup_chunks(chunk_paths)


# --- SSE streaming endpoint: emit chunk results as they finish ---

@app.post("/api/v1/transcribe/stream")
async def transcribe_stream(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    total_start = time.perf_counter()
    temp_path, chunk_paths, audio_duration, file_size_mb = await _process_file(file)
    num_chunks = len(chunk_paths)
    logger.info("[stream] Audio: %.1fs, chunks: %d (%.0fs each)", audio_duration, num_chunks, CHUNK_SECONDS)

    async def event_generator():
        yield {"event": "info", "data": json.dumps({
            "file_name": file.filename,
            "file_size_mb": file_size_mb,
            "audio_length_display": _fmt(audio_duration),
            "num_chunks": num_chunks,
        })}

        completed: dict[int, str] = {}
        next_to_emit = 0

        async def run_chunk(client, idx):
            backend = BACKENDS[idx % len(BACKENDS)]
            gpu_id = idx % len(settings.devices)  # or len(whisper_service._devices) if using WhisperService directly
            logger.info(f"Sending chunk {idx} to backend {backend} using GPU {gpu_id}...")
            # gpu_id = idx % num_gpus 
            # return await _send_chunk(client, backend, chunk_paths[idx], idx)
            result = await _send_chunk(client, backend, chunk_paths[idx], idx, gpu_id)
            return result

        try:
            async with httpx.AsyncClient(timeout=600.0) as client:
                pending = {
                    asyncio.create_task(run_chunk(client, i)): i
                    for i in range(num_chunks)
                }

                while pending:
                    done, _ = await asyncio.wait(pending, return_when=asyncio.FIRST_COMPLETED)
                    for task in done:
                        result = task.result()
                        idx = result["index"]
                        completed[idx] = result.get("text", "").strip()
                        del pending[task]

                        # Emit chunks in order as they become available
                        while next_to_emit in completed:
                            yield {"event": "chunk", "data": json.dumps({
                                "index": next_to_emit,
                                "text": completed[next_to_emit],
                            })}
                            next_to_emit += 1

            total_time = time.perf_counter() - total_start
            combined = " ".join(completed[i] for i in range(num_chunks) if completed.get(i))

            yield {"event": "done", "data": json.dumps({
                "text": combined,
                "total_processing_time_display": _fmt(total_time),
                "speed_factor": round(audio_duration / total_time, 2) if total_time > 0 else 0,
            })}

        finally:
            try:
                os.remove(temp_path)
            except OSError:
                pass
            cleanup_chunks(chunk_paths)

    return EventSourceResponse(event_generator())
