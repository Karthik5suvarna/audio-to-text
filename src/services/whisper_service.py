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
        """Resolve devices from settings. CTranslate2 expects 'cuda' or 'cpu' (no device indices)."""
        # Use settings.devices (list) if available, otherwise fall back to settings.device
        devices = settings.devices if isinstance(settings.devices, list) and settings.devices else [settings.device]
        
        # Normalize devices: strip indices like :0, :1 to get just 'cuda' or 'cpu'
        normalized = []
        for dev in devices:
            if isinstance(dev, str):
                # Strip device indices: "cuda:0" -> "cuda", "cuda:1" -> "cuda"
                base_dev = dev.split(":")[0]
                
                if base_dev == "auto":
                    try:
                        import torch
                        base_dev = "cuda" if torch.cuda.is_available() else "cpu"
                    except ImportError:
                        base_dev = "cpu"
                
                normalized.append(base_dev)
        
        # Determine compute types based on device
        compute_types = []
        for dev in normalized:
            if dev == "cuda":
                compute_types.append("float16")
            else:
                compute_types.append("int8")
        
        if not normalized:
            normalized = ["cpu"]
            compute_types = ["int8"]
        
        return normalized, compute_types

    def load_model(self):
        start = time.perf_counter()
        self._devices, self._compute_types = self._resolve_devices_and_compute()

        # Load model on the first device only (each backend loads one model)
        device = self._devices[0]
        compute_type = self._compute_types[0]
        
        logger.info(
            "Loading faster-whisper model '%s' on device=%s compute_type=%s",
            settings.model_id, device, compute_type,
        )

        self.model = WhisperModel(
            settings.model_id,
            device=device,
            compute_type=compute_type,
        )

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

        text = " ".join(seg.text.strip() for seg in segments)
        inference_time = round(time.perf_counter() - start, 3)

        return {
            "text": text,
            "inference_time_seconds": inference_time,
            "language": info.language,
            "language_probability": round(info.language_probability, 3),
            "vad_active": True,
        }

    @property
    def device_info(self) -> str:
        if self._devices == ["cpu"] and self._compute_types == ["int8"]:
            return "cpu (int8)"
        return f"{self._devices} ({self._compute_types})" if self._devices else "not loaded"


whisper_service = WhisperService()

