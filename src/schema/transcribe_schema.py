from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    text: str
    file_name: str
    file_size_mb: float
    audio_duration_seconds: float
    inference_time_seconds: float
    model_load_time_seconds: float
    device: str
