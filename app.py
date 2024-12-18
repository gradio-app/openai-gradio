import gradio as gr
import openai_gradio

gr.load(
    name='gpt-4o-mini-realtime-preview-2024-12-17',
    src=openai_gradio.registry
).launch()