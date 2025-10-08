import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    """Configuración de STT providers y LiveKit"""
       
    # Deepgram (alternativa recomendada)
    DEEPGRAM_API_KEY = os.getenv("DEEPGRAM_API_KEY")
    
    # LiveKit
    LIVEKIT_URL = os.getenv("LIVEKIT_URL", "ws://localhost:7880")
    LIVEKIT_API_KEY = os.getenv("LIVEKIT_API_KEY", "devkey")
    LIVEKIT_API_SECRET = os.getenv("LIVEKIT_API_SECRET", "secret")
    
    # Parámetros de audio
    SAMPLE_RATE = 16000
    CHANNELS = 1
    
    # Proveedores disponibles
    PROVEEDORES = {
        "Deepgram": "deepgram",
    }
    
    # Idiomas disponibles
    IDIOMAS = {
        "Español": "es",
        "English": "en",
        "Français": "fr",
        "Deutsch": "de",
        "Italiano": "it",
        "Português": "pt"
    }
    
    @classmethod
    def validar_configuracion(cls, proveedor="deepgram"):
        """Valida que las variables necesarias estén configuradas"""
        if proveedor == "deepgram":
            if not cls.DEEPGRAM_API_KEY:
                raise ValueError("DEEPGRAM_API_KEY no está configurada")
        elif proveedor == "azure":
            if not cls.AZURE_SPEECH_KEY or not cls.AZURE_SPEECH_REGION:
                raise ValueError("AZURE_SPEECH_KEY y AZURE_SPEECH_REGION deben estar configuradas")
        return True