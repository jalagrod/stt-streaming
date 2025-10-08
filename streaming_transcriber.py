import asyncio
import contextlib
import time
from dataclasses import dataclass
from typing import AsyncGenerator, Optional, Tuple

import numpy as np
import sounddevice as sd

from livekit import rtc
from livekit.agents.stt import SpeechEvent, SpeechEventType, SpeechStream
from livekit.plugins import deepgram

import aiohttp

from config import Config


@dataclass
class TranscriptionUpdate:
    """Small container used to stream updates back to the UI."""

    text: str
    info: str


class StreamingTranscriber:
    """Manage local microphone capture and Deepgram streaming STT."""

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

        self._interim_text = ""
        self._final_text = ""

    @property
    def is_running(self) -> bool:
        return self._input_stream is not None and not self._stop_requested

    @property
    def current_text(self) -> str:
        interim = self._interim_text.strip()
        final = self._final_text.strip()
        if interim:
            return f"{final} {interim}".strip()
        return final

    def status_message(self, headline: str) -> str:
        error = f"\nError: {self._error_message}" if self._error_message else ""
        status = (
            f"{headline}\n"
            f"Modelo: {self.MODEL_CODE}\n"
            f"Audio enviado: {self._audio_seconds:.2f}s "
            f"(Deepgram reporta {self._usage_seconds:.2f}s)\n"
            f"Fragmentos descartados: {self._dropped_chunks}"
        )
        status += f"\nUltimo evento: {self._last_status}{error}"
        return status

    async def start(self) -> None:
        if self.is_running:
            return

        Config.validar_configuracion("deepgram")

        loop = asyncio.get_running_loop()
        self._loop = loop
        self._audio_queue = asyncio.Queue(maxsize=50)
        self._stop_requested = False
        self._input_closed = False
        self._start_ts = time.time()
        self._audio_seconds = 0.0
        self._usage_seconds = 0.0
        self._dropped_chunks = 0
        self._error_message = None
        self._interim_text = ""
        self._final_text = ""
        self._last_status = "esperando audio"

        if self._http_session is not None and not self._http_session.closed:
            await self._http_session.close()

        self._http_session = aiohttp.ClientSession()
        self._stt = deepgram.STT(
            model="nova-3",
            language="multi",
            sample_rate=self.sample_rate,
            interim_results=True,
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

    async def iter_transcripts(self) -> AsyncGenerator[TranscriptionUpdate, None]:
        if self._speech_stream is None:
            raise RuntimeError("el stream aun no ha sido iniciado")

        async for event in self._speech_stream:
            update = self._handle_event(event)
            if update is not None:
                yield update

        self._last_status = "stream cerrado"

    def _handle_event(self, event: SpeechEvent) -> Optional[TranscriptionUpdate]:
        if event.type == SpeechEventType.INTERIM_TRANSCRIPT:
            text = (event.alternatives[0].text or "").strip()
            self._interim_text = text
            self._last_status = "transcripcion parcial"
            return TranscriptionUpdate(
                text=self.current_text,
                info=self.status_message("Transcripcion parcial"),
            )

        if event.type == SpeechEventType.FINAL_TRANSCRIPT:
            text = (event.alternatives[0].text or "").strip()
            if text:
                if self._final_text:
                    self._final_text += " "
                self._final_text += text
            self._interim_text = ""
            self._last_status = "transcripcion final"
            return TranscriptionUpdate(
                text=self.current_text,
                info=self.status_message("Transcripcion final"),
            )

        if event.type == SpeechEventType.START_OF_SPEECH:
            self._last_status = "inicio de habla"
            return TranscriptionUpdate(
                text=self.current_text,
                info=self.status_message("Habla detectada"),
            )

        if event.type == SpeechEventType.END_OF_SPEECH:
            self._last_status = "fin de habla"
            return TranscriptionUpdate(
                text=self.current_text,
                info=self.status_message("Silencio detectado"),
            )

        if event.type == SpeechEventType.PREFLIGHT_TRANSCRIPT:
            text = (event.alternatives[0].text or "").strip()
            self._interim_text = text
            self._last_status = "transcripcion estable"
            return TranscriptionUpdate(
                text=self.current_text,
                info=self.status_message("Transcripcion estable"),
            )

        if event.type == SpeechEventType.RECOGNITION_USAGE and event.recognition_usage:
            self._usage_seconds = max(
                self._usage_seconds, float(event.recognition_usage.audio_duration)
            )
            self._last_status = "metricas actualizadas"
            return TranscriptionUpdate(
                text=self.current_text,
                info=self.status_message("Metricas de Deepgram"),
            )

        return None

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
        if self._session is not None:
            with contextlib.suppress(Exception):
                await self._session.aclose()
        self._session = None
