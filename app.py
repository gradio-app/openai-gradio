import gradio as gr
import openai_gradio

gr.load(
    name='gpt-4o-2024-11-20',
    src=openai_gradio.registry,
).launch()