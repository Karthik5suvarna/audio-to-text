import logging
import time

import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.core.config import settings

logger = logging.getLogger(__name__)


class WhisperService:
    def __init__(self):
        self.pipe = None
        self.model_load_time: float = 0.0
        self._device: str = ""
        self._dtype = None

    def _resolve_device(self) -> str:
        if settings.device != "auto":
            return settings.device
        if torch.cuda.is_available():
            return "cuda:0"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def _resolve_dtype(self, device: str):
        if "cuda" in device:
            return torch.float16
        return torch.float32

    def load_model(self):
        start = time.perf_counter()

        self._device = self._resolve_device()
        self._dtype = self._resolve_dtype(self._device)

        logger.info("Loading model '%s' on device='%s' dtype=%s", settings.model_id, self._device, self._dtype)

        model = AutoModelForSpeechSeq2Seq.from_pretrained(
            settings.model_id,
            torch_dtype=self._dtype,
            low_cpu_mem_usage=True,
            use_safetensors=True,
        )
        model.to(self._device)

        processor = AutoProcessor.from_pretrained(settings.model_id)

        self.pipe = pipeline(
            "automatic-speech-recognition",
            model=model,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            torch_dtype=self._dtype,
            device=self._device,
        )

        self.model_load_time = round(time.perf_counter() - start, 3)
        logger.info("Model loaded in %.3fs", self.model_load_time)

    def transcribe(self, audio_path: str) -> dict:
        if self.pipe is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start = time.perf_counter()

        result = self.pipe(
            audio_path,
            chunk_length_s=30,
            generate_kwargs={"language": "english"},
            return_timestamps=False,
            ignore_warning=True,  # suppress experimental chunk_length_s warning
        )

        inference_time = round(time.perf_counter() - start, 3)

        return {
            "text": result["text"].strip(),
            "inference_time_seconds": inference_time,
        }

    @property
    def device_info(self) -> str:
        return self._device or "not loaded"


whisper_service = WhisperService()
