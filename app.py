import gradio as gr
import openai_gradio

gr.load(
    name='gpt-3.5-turbo',
    src=openai_gradio.registry,
).launch()