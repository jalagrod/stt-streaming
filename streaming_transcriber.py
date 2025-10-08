import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Dict, List, Optional, Tuple

import aiohttp
import numpy as np
import sounddevice as sd
from livekit import rtc
from livekit.agents.stt import SpeechEvent, SpeechEventType, SpeechStream
from livekit.plugins import deepgram

from config import Config


@dataclass
class TranscriptionUpdate:
    """Payload returned to the UI with every STT event."""

    text: str
    info: str
    conversation: List[Dict[str, str]]


class StreamingTranscriber:
    """Capture microphone audio and stream it to Deepgram with multi-speaker support."""

    MODEL_CODE = "deepgram/nova-3:multi"

    def __init__(
        self,
        *,
        sample_rate: int = Config.SAMPLE_RATE,
        channels: int = Config.CHANNELS,
        block_size: int = 320,
    ) -> None:
        self.sample_rate = sample_rate
        self.channels = channels
        self.block_size = block_size

        self._http_session: Optional[aiohttp.ClientSession] = None
        self._stt: Optional[deepgram.STT] = None
        self._speech_stream: Optional[SpeechStream] = None
        self._input_stream: Optional[sd.InputStream] = None
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._audio_queue: Optional[asyncio.Queue[Tuple[bytes, int]]] = None
        self._audio_task: Optional[asyncio.Task[None]] = None

        self._stop_requested = False
        self._input_closed = False
        self._start_ts: Optional[float] = None
        self._audio_seconds = 0.0
        self._usage_seconds = 0.0
        self._dropped_chunks = 0
        self._last_status = "idle"
        self._error_message: Optional[str] = None

        self._conversation: List[Dict[str, str]] = []
        self._interim_by_speaker: Dict[str, str] = {}
        self._speaker_alias: Dict[str, str] = {}
        self._next_speaker_index = 1

    # ------------------------------------------------------------------ #
    # Properties
    # ------------------------------------------------------------------ #
    @property
    def is_running(self) -> bool:
        return self._input_stream is not None and not self._stop_requested

    @property
    def current_text(self) -> str:
        return self._build_display_text()

    @property
    def conversation(self) -> List[Dict[str, str]]:
        return [segment.copy() for segment in self._conversation]

    # ------------------------------------------------------------------ #
    # Lifecycle management
    # ------------------------------------------------------------------ #
    def status_message(self, headline: str) -> str:
        error = f"\nError: {self._error_message}" if self._error_message else ""
        status = (
            f"{headline}\n"
            f"Modelo: {self.MODEL_CODE}\n"
            f"Audio enviado: {self._audio_seconds:.2f}s "
            f"(Deepgram reporta {self._usage_seconds:.2f}s)\n"
            f"Fragmentos descartados: {self._dropped_chunks}\n"
            f"Segmentos finales: {len(self._conversation)}"
        )
        status += f"\nUltimo evento: {self._last_status}{error}"
        return status

    async def start(self) -> None:
        if self.is_running:
            return

        Config.validar_configuracion("deepgram")

        self._loop = asyncio.get_running_loop()
        self._audio_queue = asyncio.Queue(maxsize=50)
        self._stop_requested = False
        self._input_closed = False
        self._start_ts = time.time()
        self._audio_seconds = 0.0
        self._usage_seconds = 0.0
        self._dropped_chunks = 0
        self._error_message = None
        self._conversation.clear()
        self._interim_by_speaker.clear()
        self._speaker_alias.clear()
        self._next_speaker_index = 1
        self._last_status = "esperando audio"

        if self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()

        self._http_session = aiohttp.ClientSession()
        self._stt = deepgram.STT(
            model="nova-3",
            language="multi",
            sample_rate=self.sample_rate,
            interim_results=True,
            enable_diarization=True,
            endpointing_ms=Config.DEEPGRAM_ENDPOINTING_MS,
            api_key=Config.DEEPGRAM_API_KEY,
            http_session=self._http_session,
        )
        self._speech_stream = self._stt.stream()
        self._audio_task = asyncio.create_task(self._consume_audio_queue())

        try:
            self._input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=self.channels,
                dtype="int16",
                blocksize=self.block_size,
                callback=self._on_audio_frame,
            )
            self._input_stream.start()
        except Exception as exc:
            await self._cleanup_after_failure()
            raise RuntimeError(f"No se pudo abrir el microfono: {exc}") from exc

    async def stop(self) -> None:
        if self._stop_requested:
            return

        self._stop_requested = True

        if self._input_stream is not None:
            with contextlib.suppress(Exception):
                self._input_stream.stop()
                self._input_stream.close()
            self._input_stream = None

        if self._audio_task is not None and asyncio.current_task() is not self._audio_task:
            self._audio_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._audio_task
        self._audio_task = None

        if self._speech_stream is not None and not self._input_closed:
            self._speech_stream.end_input()
            self._input_closed = True

        self._audio_queue = None

    async def aclose(self) -> None:
        await self.stop()
        if self._speech_stream is not None:
            with contextlib.suppress(Exception):
                await self._speech_stream.aclose()
        self._speech_stream = None
        self._stt = None
        if self._http_session is not None:
            with contextlib.suppress(Exception):
                await self._http_session.close()
        self._http_session = None
        self._audio_queue = None
        self._loop = None

    # ------------------------------------------------------------------ #
    # Streaming helpers
    # ------------------------------------------------------------------ #
    async def iter_transcripts(self) -> AsyncGenerator[TranscriptionUpdate, None]:
        if self._speech_stream is None:
            raise RuntimeError("el stream aun no ha sido iniciado")

        async for event in self._speech_stream:
            update = self._handle_event(event)
            if update is not None:
                yield update

        self._last_status = "stream cerrado"

    def _handle_event(self, event: SpeechEvent) -> Optional[TranscriptionUpdate]:
        if not event.alternatives:
            return None

        text = (event.alternatives[0].text or "").strip()
        speaker_id = self._extract_speaker(event)

        if event.type == SpeechEventType.INTERIM_TRANSCRIPT:
            if text and speaker_id is not None:
                self._interim_by_speaker[speaker_id] = text
            self._last_status = "transcripcion parcial"
            return self._make_update("Transcripcion parcial")

        if event.type == SpeechEventType.PREFLIGHT_TRANSCRIPT:
            if text and speaker_id is not None:
                self._interim_by_speaker[speaker_id] = text
            self._last_status = "transcripcion estable"
            return self._make_update("Transcripcion estable")

        if event.type == SpeechEventType.FINAL_TRANSCRIPT:
            if speaker_id is not None:
                self._interim_by_speaker.pop(speaker_id, None)
            if text:
                alias = self._alias_for_speaker(speaker_id)
                self._conversation.append(
                    {
                        "speaker": alias,
                        "text": text,
                        "timestamp": time.strftime("%H:%M:%S"),
                    }
                )
            self._last_status = "transcripcion final"
            return self._make_update("Transcripcion final")

        if event.type == SpeechEventType.START_OF_SPEECH:
            self._last_status = "inicio de habla"
            return self._make_update("Habla detectada")

        if event.type == SpeechEventType.END_OF_SPEECH:
            self._last_status = "fin de habla"
            return self._make_update("Silencio detectado")

        if event.type == SpeechEventType.RECOGNITION_USAGE and event.recognition_usage:
            self._usage_seconds = max(
                self._usage_seconds, float(event.recognition_usage.audio_duration)
            )
            self._last_status = "metricas actualizadas"
            return self._make_update("Metricas de Deepgram")

        return None

    def _make_update(self, headline: str) -> TranscriptionUpdate:
        return TranscriptionUpdate(
            text=self._build_display_text(),
            info=self.status_message(headline),
            conversation=self.conversation,
        )

    async def _consume_audio_queue(self) -> None:
        if self._audio_queue is None or self._speech_stream is None:
            return

        try:
            while True:
                data, frames = await self._audio_queue.get()
                frame = rtc.AudioFrame(
                    data=data,
                    sample_rate=self.sample_rate,
                    samples_per_channel=frames,
                    num_channels=self.channels,
                )
                try:
                    self._speech_stream.push_frame(frame)
                except Exception as exc:
                    self._error_message = str(exc)
                    await self.stop()
                    break

                self._audio_seconds += frames / float(self.sample_rate)
        except asyncio.CancelledError:
            pass

    def _on_audio_frame(self, indata: np.ndarray, frames: int, *_args) -> None:
        if self._stop_requested or self._audio_queue is None:
            return

        data = np.copy(indata[:, 0] if indata.ndim > 1 else indata).astype(np.int16, copy=False)
        payload = (data.tobytes(), frames)

        def _enqueue() -> None:
            if self._audio_queue is None:
                return
            try:
                self._audio_queue.put_nowait(payload)
            except asyncio.QueueFull:
                self._dropped_chunks += 1

        if self._loop is not None:
            self._loop.call_soon_threadsafe(_enqueue)

    async def _cleanup_after_failure(self) -> None:
        if self._audio_task is not None:
            self._audio_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await self._audio_task
        self._audio_task = None

        if self._speech_stream is not None:
            with contextlib.suppress(Exception):
                await self._speech_stream.aclose()
        self._speech_stream = None
        self._stt = None
        self._audio_queue = None
        if self._http_session is not None:
            with contextlib.suppress(Exception):
                await self._http_session.close()
        self._http_session = None

    def _extract_speaker(self, event: SpeechEvent) -> Optional[str]:
        speaker_id = event.alternatives[0].speaker_id if event.alternatives else None
        if speaker_id is None:
            return "__unknown__"
        return str(speaker_id)

    def _alias_for_speaker(self, speaker_id: Optional[str]) -> str:
        if not speaker_id or speaker_id == "__unknown__":
            return "Orador"
        alias = self._speaker_alias.get(speaker_id)
        if alias is None:
            alias = f"Orador {self._next_speaker_index}"
            self._next_speaker_index += 1
            self._speaker_alias[speaker_id] = alias
        return alias

    def _build_display_text(self) -> str:
        lines: List[str] = []
        for segment in self._conversation:
            lines.append(f"{segment['timestamp']} - {segment['speaker']}: {segment['text']}")

        for speaker_id, text in self._interim_by_speaker.items():
            alias = self._alias_for_speaker(speaker_id)
            lines.append(f"{time.strftime('%H:%M:%S')} - {alias} (parcial): {text}")

        return "\n".join(lines)
