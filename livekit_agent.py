import asyncio
from dotenv import load_dotenv
from livekit import agents, rtc
from livekit.agents.stt import SpeechEventType, SpeechEvent
from typing import AsyncIterable
from livekit.plugins import deepgram, azure
from config import Config

load_dotenv()

class TranscriptionAgent:
    """Agente de transcripción que procesa audio tracks"""
    
    def __init__(self, proveedor="deepgram", idioma="es"):
        self.proveedor = proveedor
        self.idioma = idioma
        self.transcripciones = []
        
        Config.validar_configuracion(proveedor)
    
    def crear_stt(self):
        """Crea instancia de STT según el proveedor configurado"""
        if self.proveedor == "deepgram":
            # Deepgram - mejor para tiempo real
            return deepgram.STT(
                model="nova-2",
                language=self.idioma
            )
        elif self.proveedor == "azure":
            # Azure Speech Services
            return azure.STT(
                speech_key=Config.AZURE_SPEECH_KEY,
                speech_region=Config.AZURE_SPEECH_REGION,
                language=self.idioma
            )
        else:
            raise ValueError(f"Proveedor no soportado: {self.proveedor}")
    
    async def process_stt_stream(self, stream: AsyncIterable[SpeechEvent]):
        """Procesa el stream de eventos de transcripción"""
        try:
            async for event in stream:
                if event.type == SpeechEventType.FINAL_TRANSCRIPT:
                    texto = event.alternatives[0].text
                    print(f"[FINAL] {texto}")
                    self.transcripciones.append({
                        "tipo": "final",
                        "texto": texto
                    })
                    
                elif event.type == SpeechEventType.INTERIM_TRANSCRIPT:
                    texto = event.alternatives[0].text
                    print(f"[INTERIM] {texto}")
                    
                elif event.type == SpeechEventType.START_OF_SPEECH:
                    print("[EVENTO] Inicio de habla detectado")
                    
                elif event.type == SpeechEventType.END_OF_SPEECH:
                    print("[EVENTO] Fin de habla detectado")
        finally:
            await stream.aclose()
    
    async def process_track(self, track: rtc.RemoteTrack):
        """Procesa un track de audio remoto"""
        print(f"Procesando track: {track.name}")
        
        # Crear STT y streams
        stt = self.crear_stt()
        stt_stream = stt.stream()
        audio_stream = rtc.AudioStream(track)
        
        async with asyncio.TaskGroup() as tg:
            # Tarea para procesar transcripciones
            stt_task = tg.create_task(self.process_stt_stream(stt_stream))
            
            # Procesar stream de audio
            async for audio_event in audio_stream:
                stt_stream.push_frame(audio_event.frame)
            
            # Indicar fin del stream de audio
            stt_stream.end_input()
            
            # Esperar a que termine el procesamiento
            await stt_task
        
        print(f"Track procesado. Total transcripciones: {len(self.transcripciones)}")
        return self.transcripciones


async def entrypoint(ctx: agents.JobContext):
    """Punto de entrada del agente LiveKit"""
    
    # Configurar proveedor e idioma
    proveedor = "deepgram"  # o "azure"
    idioma = "es"
    
    agent = TranscriptionAgent(proveedor=proveedor, idioma=idioma)
    
    @ctx.room.on("track_subscribed")
    def on_track_subscribed(track: rtc.RemoteTrack):
        print(f"Subscrito a track: {track.name}")
        asyncio.create_task(agent.process_track(track))
    
    print(f"Agente iniciado - Proveedor: {proveedor}, Idioma: {idioma}")
    print("Esperando conexiones...")


if __name__ == "__main__":
    agents.cli.run_app(agents.WorkerOptions(entrypoint_fnc=entrypoint))
