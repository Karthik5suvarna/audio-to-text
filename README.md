# Audio to Text — faster-whisper

Local audio transcription using [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (CTranslate2 backend) on GPU.

POST an audio file to the API; get back the transcribed text with timing metrics. Includes a chunk-parallel orchestrator for distributing work across multiple Whisper backends.

## Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** with drivers installed (or CPU with int8 quantization for dev)
- **ffmpeg** — required for audio decoding and chunk splitting

```bash
# Ubuntu
sudo apt update && sudo apt install ffmpeg -y

# macOS
brew install ffmpeg
```

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> **Note:** `faster-whisper` uses CTranslate2 which supports **CUDA** and **CPU**. MPS (Apple Silicon) is not supported — it falls back to CPU with int8 quantization, which is still faster than the old HuggingFace transformers pipeline on MPS.

## Run (single backend)

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

First startup downloads the model from HuggingFace. Use **http://localhost:8000/docs** for Swagger UI.

## Run (chunk-parallel with multiple backends)

One audio file is **split into chunks**, each chunk is sent to a **different backend in parallel**, transcripts are **merged in order**.

**1. Start backends** (one terminal each):

```bash
# Ubuntu multi-GPU — one backend per GPU
DEVICE=cuda:0 python -m uvicorn src.main:app --host 127.0.0.1 --port 8001
DEVICE=cuda:1 python -m uvicorn src.main:app --host 127.0.0.1 --port 8002
DEVICE=cuda:2 python -m uvicorn src.main:app --host 127.0.0.1 --port 8003

# Mac / single GPU (CPU int8 for each)
python -m uvicorn src.main:app --host 127.0.0.1 --port 8001
python -m uvicorn src.main:app --host 127.0.0.1 --port 8002
python -m uvicorn src.main:app --host 127.0.0.1 --port 8003
```

**2. Start the orchestrator** (port 8000):

```bash
python -m uvicorn src.load_balancer:app --host 0.0.0.0 --port 8000
```

**3. Send a request:**

```bash
# Standard (wait for full result)
curl -X POST http://localhost:8000/api/v1/transcribe -F "file=@audio.mp4"

# Streaming (SSE — partial results as chunks finish)
curl -N -X POST http://localhost:8000/api/v1/transcribe/stream -F "file=@audio.mp4"
```

## Orchestrator config (env vars)

| Variable | Default | Description |
|----------|---------|-------------|
| `BACKEND_URLS` | `http://127.0.0.1:8001,...:8003` | Comma-separated backend URLs |
| `CHUNK_SECONDS` | `60` | Chunk length in seconds (60–120 recommended) |
| `CONCURRENCY_PER_BACKEND` | `1` | Max concurrent requests per backend |

## Key optimizations

| # | Optimization | Implementation |
|---|-------------|----------------|
| 1 | **Bigger chunks** (60s default) | Fewer chunks = less overhead |
| 2 | **Per-backend concurrency control** | Semaphore prevents overwhelming a backend |
| 3 | **WAV/PCM chunks at 16kHz** | No re-encode to lossy; fast to write/read |
| 4 | **File handle streaming** | Chunks sent via file handle, not loaded into memory |
| 5 | **VAD filtering** | Silences skipped — less compute for meetings/podcasts |
| 6 | **Greedy decoding** | `beam_size=1, best_of=1` — fastest decode |
| 7 | **faster-whisper + CTranslate2** | 4–8x faster than HuggingFace transformers pipeline |
| 8 | **Controlled chunk feeding** | Semaphore queues; next chunk sent when backend is free |
| 9 | **No overlap** | Zero overlap for max speed (add later if boundary issues) |
| 10 | **SSE streaming** | `POST /api/v1/transcribe/stream` returns partial results as they complete |

## API

### `POST /api/v1/transcribe` (single backend)

```json
{
  "text": "transcribed text...",
  "file_name": "recording.wav",
  "file_size_mb": 5.23,
  "speed_factor": 65.2,
  "audio_length_display": "2.0 min",
  "transcription_time_display": "1.8 sec",
  "model_load_display": "8.4 sec",
  "hardware_display": "NVIDIA GPU (CUDA)"
}
```

### `POST /api/v1/transcribe` (via orchestrator)

```json
{
  "text": "combined transcribed text...",
  "file_name": "meeting.mp4",
  "file_size_mb": 8.95,
  "audio_length_display": "25.6 min",
  "total_processing_time_display": "2.1 min",
  "speed_factor": 12.2,
  "num_chunks": 26,
  "chunk_seconds": 60,
  "backends_used": 3
}
```

### `POST /api/v1/transcribe/stream` (SSE)

Returns Server-Sent Events:

```
event: info
data: {"file_name": "audio.mp4", "num_chunks": 3, ...}

event: chunk
data: {"index": 0, "text": "first chunk transcript..."}

event: chunk
data: {"index": 1, "text": "second chunk transcript..."}

event: done
data: {"text": "full combined text...", "speed_factor": 12.2, ...}
```

### cURL example

```bash
curl -X POST http://localhost:8000/api/v1/transcribe -F "file=@recording.wav"
```

## Project Structure

```
src/
├── main.py                          # FastAPI app + Whisper (single instance)
├── load_balancer.py                 # Chunk-parallel orchestrator + SSE streaming
├── core/config.py                   # Settings (from .env)
├── services/
│   ├── whisper_service.py           # faster-whisper model + VAD + greedy decode
│   ├── file_service.py              # Upload handling + cleanup
│   └── audio_splitter.py            # Split audio into chunks (ffmpeg)
├── schema/transcribe_schema.py      # Response schema
└── api/v1/endpoint/
    └── transcribe.py                # POST /api/v1/transcribe
```

## Configuration (.env)

| Variable | Default | Description |
|----------|---------|-------------|
| `MODEL_ID` | `large-v3` | faster-whisper model (tiny/base/small/medium/large-v3/distil-large-v3) |
| `DEVICE` | `auto` | `auto`, `cuda`, `cuda:0`, or `cpu` |
| `UPLOAD_DIR` | `storage/uploads` | Temp directory for uploaded files |
| `MAX_FILE_SIZE_MB` | `500` | Max upload size |

## Performance notes

- **CUDA + float16:** 20–50x real-time with large-v3; the single biggest speedup
- **CPU + int8:** ~2–5x real-time depending on core count (good for dev)
- **VAD filtering:** Skips silence — can cut compute 30–60% on meetings/podcasts
- **Greedy decode:** 2–3x faster than beam search with minimal accuracy loss for English
- **Multi-GPU parallel:** Near-linear scaling (3 GPUs ≈ 3x throughput)
- **Single GPU, 3 backends:** No per-file speedup; helps with concurrent requests only
