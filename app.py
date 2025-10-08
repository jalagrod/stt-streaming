import gradio as gr

from config import Config
from streaming_transcriber import StreamingTranscriber


async def start_streaming(
    state: StreamingTranscriber | None,
) -> tuple[str, str, StreamingTranscriber | None, dict, dict]:
    """Start a Deepgram streaming session and yield UI updates."""
    if isinstance(state, StreamingTranscriber) and state.is_running:
        yield (
            state.current_text,
            state.status_message("Streaming ya estaba activo"),
            state,
            gr.update(interactive=False),
            gr.update(interactive=True),
        )
        return

    transcriber = StreamingTranscriber()

    try:
        await transcriber.start()
    except Exception as exc:
        previous_text = state.current_text if isinstance(state, StreamingTranscriber) else ""
        yield (
            previous_text,
            f"Error al iniciar el streaming: {exc}",
            state if isinstance(state, StreamingTranscriber) else None,
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
        return

    yield (
        transcriber.current_text,
        transcriber.status_message("Streaming iniciado"),
        transcriber,
        gr.update(interactive=False),
        gr.update(interactive=True),
    )

    try:
        async for update in transcriber.iter_transcripts():
            yield (
                update.text,
                update.info,
                transcriber,
                gr.update(interactive=False),
                gr.update(interactive=True),
            )
    except Exception as exc:
        await transcriber.aclose()
        yield (
            transcriber.current_text,
            f"Error durante la transcripcion: {exc}",
            None,
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
        return

    await transcriber.aclose()
    yield (
        transcriber.current_text,
        transcriber.status_message("Streaming detenido"),
        None,
        gr.update(interactive=True),
        gr.update(interactive=False),
    )


async def stop_streaming(
    state: StreamingTranscriber | None,
) -> tuple[str, str, StreamingTranscriber | None, dict, dict]:
    """Request a graceful stop for the current streaming session."""
    if not isinstance(state, StreamingTranscriber):
        return (
            "",
            "No hay streaming activo.",
            None,
            gr.update(interactive=True),
            gr.update(interactive=False),
        )

    await state.stop()
    return (
        state.current_text,
        state.status_message("Deteniendo streaming..."),
        state,
        gr.update(interactive=False),
        gr.update(interactive=False),
    )


def crear_interfaz() -> gr.Blocks:
    """Build the Gradio UI for streaming STT."""
    with gr.Blocks(theme=gr.themes.Soft(), title="Deepgram Streaming STT") as app:
        gr.Markdown(
            """
            ## Streaming STT con Deepgram nova-3:multi

            Esta demo captura el microfono local y envia audio a Deepgram usando
            LiveKit Agents. Necesitas definir `DEEPGRAM_API_KEY` en `.env`.
            """
        )

        state_holder = gr.State(None)

        transcription_output = gr.Textbox(
            label="Transcripcion",
            placeholder="El texto aparecera aqui en cuanto hables...",
            lines=14,
            autofocus=True,
        )
        status_output = gr.Textbox(
            label="Estado",
            value="Presiona Iniciar para comenzar a escuchar el microfono local.",
            lines=6,
        )

        with gr.Row():
            start_button = gr.Button("Iniciar streaming", variant="primary")
            stop_button = gr.Button("Detener", variant="secondary", interactive=False)

        start_button.click(
            fn=start_streaming,
            inputs=[state_holder],
            outputs=[transcription_output, status_output, state_holder, start_button, stop_button],
            stream_every=0.1,        # opcional; controla cada cuánto se refrescan los yields
            show_progress=False,
        )

        stop_button.click(
            fn=stop_streaming,
            inputs=[state_holder],
            outputs=[transcription_output, status_output, state_holder, start_button, stop_button],
            show_progress=False,
        )

        gr.Markdown(
            """
            ### Notas rapidas
            - Se usa `StreamingTranscriber` con `AgentSession` configurado como `deepgram/nova-3:multi`.
            - El audio se captura a 16 kHz mono para reducir latencia.
            - Asegurate de cerrar la sesion antes de iniciar otra para evitar conflictos.
            """
        )

    return app


if __name__ == "__main__":
    try:
        print("Iniciando aplicacion Gradio...")
        Config.validar_configuracion("deepgram")
        interfaz = crear_interfaz()
        interfaz.launch(server_name="127.0.0.1", server_port=8080, share=False)
    except Exception as exc:
        print(f"Error al iniciar la aplicacion: {exc}")
