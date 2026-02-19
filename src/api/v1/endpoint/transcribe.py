import asyncio

from fastapi import APIRouter, File, HTTPException, UploadFile

from src.core.config import settings
from src.schema.transcribe_schema import TranscriptionResponse
from src.services.file_service import file_service
from src.services.whisper_service import whisper_service

router = APIRouter()


@router.post("/transcribe", response_model=TranscriptionResponse)
async def transcribe_audio(file: UploadFile = File(...)):
    if not file.filename:
        raise HTTPException(status_code=400, detail="No file provided.")

    if not file_service.validate_extension(file.filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported format. Allowed: {', '.join(settings.allowed_extensions)}",
        )

    filepath = await file_service.save_upload(file)

    try:
        file_size_mb = round(file_service.get_file_size_mb(filepath), 2)
        if file_size_mb > settings.max_file_size_mb:
            raise HTTPException(
                status_code=400,
                detail=f"File too large ({file_size_mb} MB). Max allowed: {settings.max_file_size_mb} MB.",
            )

        audio_duration = file_service.get_audio_duration(filepath)

        result = await asyncio.to_thread(whisper_service.transcribe, filepath)

        inference = result["inference_time_seconds"]
        model_load = whisper_service.model_load_time
        device_raw = whisper_service.device_info

        speed_factor = round(audio_duration / inference, 2) if inference > 0 else 0.0

        def _display_time(sec: float) -> str:
            if sec >= 60:
                return f"{sec / 60:.1f} min"
            return f"{sec:.1f} sec"

        def _hardware_display(dev: str) -> str:
            if dev.startswith("cuda"):
                return "NVIDIA GPU (CUDA)"
            if dev == "mps":
                return "Apple GPU (MPS)"
            return "CPU"

        return TranscriptionResponse(
            text=result["text"],
            file_name=file.filename,
            file_size_mb=file_size_mb,
            speed_factor=speed_factor,
            audio_length_display=_display_time(audio_duration),
            transcription_time_display=_display_time(inference),
            model_load_display=_display_time(model_load),
            hardware_display=_hardware_display(device_raw),
        )
    finally:
        file_service.delete_file(filepath)
