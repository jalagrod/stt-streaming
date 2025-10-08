import gradio as gr

from config import Config
from streaming_transcriber import StreamingTranscriber


async def start_streaming(
    state: StreamingTranscriber | None,
    history: list[dict[str, str]] | None,
) -> tuple[str, str, StreamingTranscriber | None, list[dict[str, str]], dict, dict]:
    """Start or reuse a streaming STT session."""
    history = history or []

    if isinstance(state, StreamingTranscriber) and state.is_running:
        yield (
            state.current_text,
            state.status_message("Streaming ya estaba activo"),
            state,
            state.conversation,
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
            history,
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
        return

    yield (
        transcriber.current_text,
        transcriber.status_message("Streaming iniciado"),
        transcriber,
        transcriber.conversation,
        gr.update(interactive=False),
        gr.update(interactive=True),
    )

    try:
        async for update in transcriber.iter_transcripts():
            yield (
                update.text,
                update.info,
                transcriber,
                update.conversation,
                gr.update(interactive=False),
                gr.update(interactive=True),
            )
    except Exception as exc:
        error_text = transcriber.current_text
        error_conversation = transcriber.conversation
        await transcriber.aclose()
        yield (
            error_text,
            f"Error durante la transcripcion: {exc}",
            None,
            error_conversation,
            gr.update(interactive=True),
            gr.update(interactive=False),
        )
        return

    final_text = transcriber.current_text
    final_conversation = transcriber.conversation
    final_status = transcriber.status_message("Streaming detenido")
    await transcriber.aclose()
    yield (
        final_text,
        final_status,
        None,
        final_conversation,
        gr.update(interactive=True),
        gr.update(interactive=False),
    )


async def stop_streaming(
    state: StreamingTranscriber | None,
    history: list[dict[str, str]] | None,
) -> tuple[str, str, StreamingTranscriber | None, list[dict[str, str]], dict, dict]:
    """Stop the active streaming session and persist the conversation."""
    history = history or []

    if not isinstance(state, StreamingTranscriber):
        return (
            "",
            "No hay streaming activo.",
            None,
            history,
            gr.update(interactive=True),
            gr.update(interactive=False),
        )

    await state.stop()
    conversation = state.conversation
    status = state.status_message("Streaming detenido")
    transcript_text = "\n".join(
        f"{entry['timestamp']} - {entry['speaker']}: {entry['text']}"
        for entry in conversation
    )
    await state.aclose()

    return (
        transcript_text,
        status,
        None,
        conversation,
        gr.update(interactive=True),
        gr.update(interactive=False),
    )


def crear_interfaz() -> gr.Blocks:
    """Build the Gradio UI for multi-speaker streaming STT."""
    with gr.Blocks(theme=gr.themes.Soft(), title="Deepgram Streaming STT") as app:
        gr.Markdown(
            """
            ## Conversaciones en tiempo real con Deepgram nova-3:multi

            Captura múltiples voces desde tu micrófono y obtén transcripción en streaming
            con diarización básica. Asegúrate de definir `DEEPGRAM_API_KEY` en tu `.env`.
            """
        )

        state_holder = gr.State(None)
        conversation_holder = gr.State([])

        transcription_output = gr.Textbox(
            label="Conversación",
            placeholder="Las intervenciones aparecerán aquí con el nombre del orador...",
            lines=16,
        )
        status_output = gr.Textbox(
            label="Estado",
            value="Pulsa «Iniciar streaming» para comenzar a escuchar el micrófono local.",
            lines=6,
        )

        with gr.Row():
            start_button = gr.Button("Iniciar streaming", variant="primary")
            stop_button = gr.Button("Detener", variant="secondary", interactive=False)

        start_button.click(
            fn=start_streaming,
            inputs=[state_holder, conversation_holder],
            outputs=[
                transcription_output,
                status_output,
                state_holder,
                conversation_holder,
                start_button,
                stop_button,
            ],
            stream_every=0.1,
            show_progress=False,
        )

        stop_button.click(
            fn=stop_streaming,
            inputs=[state_holder, conversation_holder],
            outputs=[
                transcription_output,
                status_output,
                state_holder,
                conversation_holder,
                start_button,
                stop_button,
            ],
            show_progress=False,
        )

        gr.Markdown(
            """
            ### Notas rápidas
            - El modelo se configura como `deepgram/nova-3:multi` con diarización habilitada.
            - El historial completo queda guardado en `conversation_holder` para uso posterior.
            - Puedes copiar la transcripción final desde el cuadro “Conversación”.
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
