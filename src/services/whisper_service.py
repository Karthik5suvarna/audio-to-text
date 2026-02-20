import logging
import time

from faster_whisper import WhisperModel

from src.core.config import settings

logger = logging.getLogger(__name__)


class WhisperService:
    def __init__(self):
        self.model: WhisperModel | None = None
        self.model_load_time: float = 0.0
        # self._device: str = ""
        # self._compute_type: str = ""
        self._devices: list[str] = []  # List to hold multiple devices
        self._compute_types: list[str] = []  # List to hold compute types

    # def _resolve_device_and_compute(self) -> tuple[str, str]:
    def _resolve_devices_and_compute(self) -> tuple[list[str], list[str]]:
        dev = settings.device
        if dev == "auto":
            try:
                import torch
                if torch.cuda.is_available():
                    # dev = "cuda"
                    dev = ["cuda:0", "cuda:1"]  
                else:
                    dev = "cpu"
            except ImportError:
                dev = "cpu"

        # if dev.startswith("cuda"):
        #     return dev, "float16"
        # Handle the devices and compute types
        if isinstance(dev, list) and all(d.startswith("cuda") for d in dev):
            return dev, ["float16"] * len(dev)
        else:
            return ["cpu"], ["int8"]

        # MPS not supported by CTranslate2 — fall back to CPU with int8 for speed
        return "cpu", "int8"

    def load_model(self):
        start = time.perf_counter()
        self._device, self._compute_type = self._resolve_device_and_compute()

        logger.info(
            "Loading faster-whisper model '%s' on device='%s' compute_type='%s'",
            settings.model_id, self._device, self._compute_type,
        )

        # self.model = WhisperModel(
        #     settings.model_id,
        #     device=self._device,
        #     compute_type=self._compute_type,
        # )
        # Initialize the Whisper model for each GPU
        self.model = []
        for device, compute_type in zip(self._devices, self._compute_types):
            self.model.append(WhisperModel(
                settings.model_id,
                device=device,
                compute_type=compute_type,
            ))

        self.model_load_time = round(time.perf_counter() - start, 3)
        logger.info("Model loaded in %.3fs", self.model_load_time)

    def transcribe(self, audio_path: str) -> dict:
        if self.model is None:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start = time.perf_counter()

        segments, info = self.model.transcribe(
            audio_path,
            language="en",
            beam_size=1,
            best_of=1,
            vad_filter=True,
            vad_parameters=dict(
                min_silence_duration_ms=500,
                speech_pad_ms=200,
            ),
            without_timestamps=True,
        )

        chunk_size = len(segments) // len(self.model) 

        # Process chunks on multiple GPUs
        results = []
        for idx, model in enumerate(self.model):
            chunk_segments = segments[idx*chunk_size:(idx+1)*chunk_size]  # Assign chunks to each GPU
            text = " ".join(seg.text.strip() for seg in chunk_segments)
            results.append(text)

        # Merge results
        combined_text = " ".join(results)
        inference_time = round(time.perf_counter() - start, 3)

        return {
            "text": combined_text,
            "inference_time_seconds": inference_time,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "vad_active": True,
        } 

        # text = " ".join(seg.text.strip() for seg in segments)
        # inference_time = round(time.perf_counter() - start, 3)

        # return {
        #     "text": text,
        #     "inference_time_seconds": inference_time,
        #     "language": info.language,
        #     "language_probability": round(info.language_probability, 3),
        #     "vad_active": True,
        # }

    @property
    def device_info(self) -> str:
        if self._device == "cpu" and self._compute_type == "int8":
            return "cpu (int8)"
        return f"{self._device} ({self._compute_type})" if self._device else "not loaded"


whisper_service = WhisperService()

