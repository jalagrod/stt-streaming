import os

from dotenv import load_dotenv

load_dotenv()


class Config:
    """Application configuration values."""

    # Deepgram credentials
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")

    # Optional LiveKit values (kept for compatibility with livekit_agent.py)
    LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
    LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")

    # Audio parameters
    SAMPLE_RATE = int(os.getenv("SAMPLE_RATE", "16000"))
    CHANNELS = int(os.getenv("CHANNELS", "1"))

    @classmethod
    def validar_configuracion(cls, proveedor: str = "deepgram") -> bool:
        """Ensure required env vars are present for the given provider."""
        if proveedor == "deepgram" and not cls.DEEPGRAM_API_KEY:
            raise ValueError("DEEPGRAM_API_KEY no esta configurada")
        return True
