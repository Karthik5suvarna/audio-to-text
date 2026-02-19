import logging
import os
import signal
from contextlib import asynccontextmanager

# Reduce transformers log noise (logits processor etc.)
logging.getLogger("transformers.generation.utils").setLevel(logging.ERROR)

from fastapi import FastAPI

from src.api.v1.endpoint.transcribe import router as transcribe_router
from src.services.whisper_service import whisper_service

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(name)s | %(message)s")
logger = logging.getLogger(__name__)


def _force_exit_on_sigint(*_):
    """Exit immediately on Ctrl+C so we don't wait for in-flight transcription."""
    os._exit(0)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # First Ctrl+C exits immediately instead of waiting for transcription to finish
    signal.signal(signal.SIGINT, _force_exit_on_sigint)
    logger.info("Starting up — loading Whisper model…")
    whisper_service.load_model()
    logger.info("Model ready. Accepting requests.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="Audio to Text", version="0.1.0", lifespan=lifespan)

app.include_router(transcribe_router, prefix="/api/v1", tags=["transcription"])
