import gradio as gr
import asyncio
from config import Config

def formatear_tiempo(segundos):
    """Formatea segundos a formato legible"""
    if segundos < 60:
        return f"{segundos:.2f}s"
    minutos = int(segundos // 60)
    segs = segundos % 60
    return f"{minutos}m {segs:.1f}s"

async def transcribir_audio_async(audio_input, proveedor_nombre, idioma_nombre):
    """Función asíncrona de transcripción"""
    if audio_input is None:
        return "", "Error: No se ha proporcionado ningún audio"
    
    # Obtener códigos
    proveedor = Config.PROVEEDORES.get(proveedor_nombre, "deepgram")
    idioma = Config.IDIOMAS.get(idioma_nombre, "es")
    
    try:
        # Crear tester
        tester = LocalAudioTester(proveedor=proveedor, idioma=idioma)
        
        # Transcribir
        transcripciones = await tester.transcribir_archivo(audio_input)
        
        # Formatear resultado
        texto_final = " ".join(transcripciones)
        
        info = f"""
Transcripción completada

Proveedor: {proveedor_nombre}
Idioma: {idioma_nombre} ({idioma})
Palabras: {len(texto_final.split())}
        """
        
        return texto_final, info
        
    except Exception as e:
        return "", f"Error: {str(e)}"

def transcribir_audio(audio_input, proveedor, idioma):
    """Wrapper síncrono para Gradio"""
    return asyncio.run(transcribir_audio_async(audio_input, proveedor, idioma))

def crear_interfaz():
    """Crea la interfaz de Gradio"""
    
    with gr.Blocks(theme=gr.themes.Soft(), title="LiveKit STT Transcriptor") as app:
        
        gr.Markdown("""
        # Transcriptor de Audio con LiveKit STT
        
        Transcribe audio en tiempo real usando plugins de LiveKit.
        Soporta Deepgram y Azure Speech Services.
        """)
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### Entrada de Audio")
                
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="filepath",
                    label="Graba o sube un archivo de audio"
                )
                
                proveedor = gr.Dropdown(
                    choices=list(Config.PROVEEDORES.keys()),
                    value="Deepgram",
                    label="Proveedor STT"
                )
                
                idioma = gr.Dropdown(
                    choices=list(Config.IDIOMAS.keys()),
                    value="Español",
                    label="Idioma"
                )
                
                transcribir_btn = gr.Button("Transcribir", variant="primary", size="lg")
                
                gr.Markdown("""
                ### Requisitos
                - Deepgram: Necesitas DEEPGRAM_API_KEY en .env
                - Azure: Necesitas AZURE_SPEECH_KEY y AZURE_SPEECH_REGION
                
                ### Consejos
                - Habla claramente
                - Evita ruido de fondo
                - Deepgram es mejor para tiempo real
                """)
            
            with gr.Column(scale=2):
                gr.Markdown("### Resultado")
                
                transcripcion_output = gr.Textbox(
                    label="Transcripción",
                    placeholder="La transcripción aparecerá aquí...",
                    lines=15
                )
                
                info_output = gr.Textbox(
                    label="Información",
                    lines=5
                )
        
        transcribir_btn.click(
            fn=transcribir_audio,
            inputs=[audio_input, proveedor, idioma],
            outputs=[transcripcion_output, info_output]
        )
        
        gr.Markdown("""
        ---
        Powered by LiveKit Agents Framework
        """)
    
    return app

if __name__ == "__main__":
    try:
        print("Iniciando aplicación Gradio...")
        print("Validando configuración...")
        
        # Intentar validar al menos un proveedor
        tiene_deepgram = Config.DEEPGRAM_API_KEY is not None
        tiene_azure = Config.AZURE_SPEECH_KEY is not None and Config.AZURE_SPEECH_REGION is not None
        
        if not tiene_deepgram and not tiene_azure:
            print("\nADVERTENCIA: No hay proveedores configurados")
            print("Configura DEEPGRAM_API_KEY o AZURE_SPEECH_KEY/AZURE_SPEECH_REGION en .env")
        
        app = crear_interfaz()
        
        print("\nAplicación lista")
        print("Abriendo en el navegador...")
        
        app.launch(
            server_name="127.0.0.1",
            server_port=8080,
            share=False
        )
        
    except Exception as e:
        print(f"\nError al iniciar la aplicación: {e}")