# Audio to Text — Whisper Large V3

Local audio transcription service using [Whisper Large V3](https://huggingface.co/openai/whisper-large-v3) running on GPU.

POST an audio file to the API; get back the transcribed text plus audio duration and inference time. Interactive docs at `/docs`.

## Prerequisites

- **Python 3.10+**
- **CUDA-capable GPU** with drivers installed (or MPS on macOS for dev)
- **ffmpeg** — required by transformers for audio decoding

```bash
# Ubuntu
sudo apt update && sudo apt install ffmpeg -y

# macOS
brew install ffmpeg
```

## Setup

```bash
# 1. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate

# 2. Install all dependencies (torch + CUDA 12.1 is the default)
pip install -r requirements.txt

# 3. (Optional) Edit .env to change settings
#    MODEL_ID, DEVICE (auto/cuda:0/cpu), UPLOAD_DIR, MAX_FILE_SIZE_MB
```

If pip prompts for **Azure DevOps** (e.g. “User for pkgs.dev.azure.com”), your global pip config is using an extra index. Install using only PyPI + PyTorch for this project:

```bash
PIP_INDEX_URL=https://pypi.org/simple/ PIP_EXTRA_INDEX_URL=https://download.pytorch.org/whl/cu121 pip install -r requirements.txt
```

> **Different CUDA version?** Edit the `--extra-index-url` line in `requirements.txt` to match your setup (cu118, cu121, cu124).
>
> **macOS dev?** Replace the `--extra-index-url` line with `--extra-index-url https://download.pytorch.org/whl/cpu` or just remove it — pip will install a CPU/MPS-compatible torch.

## Whisper Large V3 model (Hugging Face)

You **do not** install the model separately. The app uses the [transformers](https://huggingface.co/docs/transformers) library, which downloads **openai/whisper-large-v3** from Hugging Face automatically:

1. **First time you run the app** — `uvicorn` starts, the code calls `from_pretrained("openai/whisper-large-v3")`, and Hugging Face downloads the model (~3 GB) and caches it on disk.
2. **Later runs** — The model is loaded from the cache; no download.

No Hugging Face account or `huggingface-cli login` is required for this model (it’s public).

- **Cache location:** `~/.cache/huggingface/hub/` (override with env var `HF_HOME` or `TRANSFORMERS_CACHE` if you want).
- **Optional pre-download:** To download the model before starting the server (e.g. on a slow connection), run once:
  ```bash
  python -c "from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor; AutoModelForSpeechSeq2Seq.from_pretrained('openai/whisper-large-v3'); AutoProcessor.from_pretrained('openai/whisper-large-v3')"
  ```

## Run

From the project root, use the **venv’s** Python so all dependencies (including `aiofiles`) are found:

```bash
.venv/bin/python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

Or if your shell is already activated (`source .venv/bin/activate`), run:

```bash
python -m uvicorn src.main:app --host 0.0.0.0 --port 8000
```

On first startup the model will be downloaded from Hugging Face (if not already cached). Use **http://localhost:8000/docs** for Swagger UI to try the API.

## API

### `POST /api/v1/transcribe`

Multipart file upload. Returns JSON:

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

### cURL example

```bash
curl -X POST http://localhost:8000/api/v1/transcribe \
  -F "file=@recording.wav"
```

## Project Structure

```
src/
├── main.py                          # FastAPI app + lifespan
├── core/config.py                   # Settings (from .env)
├── services/
│   ├── whisper_service.py           # Model loading + inference
│   └── file_service.py              # Upload handling + cleanup
├── schema/transcribe_schema.py      # Response schema
└── api/v1/endpoint/
    └── transcribe.py                # POST /api/v1/transcribe
```

## Configuration (.env)

| Variable          | Default                  | Description                              |
| ----------------- | ------------------------ | ---------------------------------------- |
| `MODEL_ID`        | `openai/whisper-large-v3`| HuggingFace model identifier             |
| `DEVICE`          | `auto`                   | `auto`, `cuda:0`, `mps`, or `cpu`        |
| `UPLOAD_DIR`      | `storage/uploads`        | Temp directory for uploaded files         |
| `MAX_FILE_SIZE_MB`| `500`                    | Max upload size (for future validation)   |
