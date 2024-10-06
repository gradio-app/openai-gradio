import gradio as gr
import openai_gradio

with gr.Blocks() as demo:
    with gr.Tab("GPT-4-turbo"):
        gr.load('gpt-4-turbo', src=openai_gradio.registry)
    with gr.Tab("GPT-3.5-turbo"):
        gr.load('gpt-3.5-turbo', src=openai_gradio.registry)

demo.launch()