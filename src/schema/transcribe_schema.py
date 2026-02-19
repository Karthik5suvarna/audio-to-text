from pydantic import BaseModel


class TranscriptionResponse(BaseModel):
    text: str
    file_name: str
    file_size_mb: float
    speed_factor: float
    audio_length_display: str
    transcription_time_display: str
    model_load_display: str
    hardware_display: str
