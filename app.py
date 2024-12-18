import gradio as gr
import openai_gradio

gr.load(
    name='gpt-4o-realtime-preview-2024-10-01',
    src=openai_gradio.registry
).launch()