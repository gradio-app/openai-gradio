import os
from openai import OpenAI
import gradio as gr
from typing import Callable
import base64
import asyncio
from threading import Event, Thread
import numpy as np
from gradio_webrtc import (
    AdditionalOutputs,
    StreamHandler,
    WebRTC,
    get_twilio_turn_credentials,
)
from pydub import AudioSegment
import time

__version__ = "0.0.5"

SAMPLE_RATE = 24000

def encode_audio(sample_rate, data):
    segment = AudioSegment(
        data.tobytes(),
        frame_rate=sample_rate,
        sample_width=data.dtype.itemsize,
        channels=1,
    )
    pcm_audio = (
        segment.set_frame_rate(SAMPLE_RATE).set_channels(1).set_sample_width(2).raw_data
    )
    return base64.b64encode(pcm_audio).decode("utf-8")

class OpenAIHandler(StreamHandler):
    def __init__(
        self,
        expected_layout="mono",
        output_sample_rate=SAMPLE_RATE,
        output_frame_size=480,
    ) -> None:
        super().__init__(
            expected_layout,
            output_sample_rate,
            output_frame_size,
            input_sample_rate=SAMPLE_RATE,
        )
        self.connection = None
        self.all_output_data = None
        self.args_set = Event()
        self.quit = Event()
        self.connected = Event()
        self.thread = None
        self._generator = None
        self.last_frame_time = time.time()
        self.TIMEOUT_THRESHOLD = 60

    def copy(self):
        return OpenAIHandler(
            expected_layout=self.expected_layout,
            output_sample_rate=self.output_sample_rate,
            output_frame_size=self.output_frame_size,
        )

    def _initialize_connection(self, api_key: str):
        """Connect to realtime API. Run forever in separate thread to keep connection open."""
        self.client = OpenAI(api_key=api_key)
        with self.client.beta.realtime.connect(
            model="gpt-4o-realtime-preview-2024-10-01"
        ) as conn:
            conn.session.update(session={"turn_detection": {"type": "server_vad"}})
            self.connection = conn
            self.connected.set()
            while not self.quit.is_set():
                time.sleep(0.25)

    async def fetch_args(self):
        if self.channel:
            self.channel.send("tick")

    def set_args(self, args):
        super().set_args(args)
        self.args_set.set()

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        current_time = time.time()
        if current_time - self.last_frame_time > self.TIMEOUT_THRESHOLD:
            if self.connection:
                self.connection.close()
                self.connection = None
            self.last_frame_time = current_time
            return

        if not self.channel:
            return
        if not self.connection:
            asyncio.run_coroutine_threadsafe(self.fetch_args(), self.loop)
            self.args_set.wait()
            self.thread = Thread(
                target=self._initialize_connection, args=(self.latest_args[-1],)
            )
            self.thread.start()
            self.connected.wait()
        try:
            assert self.connection, "Connection not initialized"
            sample_rate, array = frame
            array = array.squeeze()
            audio_message = encode_audio(sample_rate, array)
            self.connection.input_audio_buffer.append(audio=audio_message)
            self.last_frame_time = current_time
        except Exception as e:
            print(f"Error in receive: {str(e)}")
            import traceback
            traceback.print_exc()

    def generator(self):
        while True:
            if not self.connection:
                yield None
                continue
            for event in self.connection:
                if event.type == "response.audio_transcript.done":
                    yield AdditionalOutputs(event)
                if event.type == "response.audio.delta":
                    yield (
                        self.output_sample_rate,
                        np.frombuffer(
                            base64.b64decode(event.delta), dtype=np.int16
                        ).reshape(1, -1),
                    )

    def emit(self) -> tuple[int, np.ndarray] | None:
        if not self.connection:
            return None
        if not self._generator:
            self._generator = self.generator()
        try:
            return next(self._generator)
        except StopIteration:
            self._generator = self.generator()
            return None

    def shutdown(self) -> None:
        if self.connection:
            self.connection.close()
            self.quit.set()
            if self.thread:
                self.thread.join(timeout=5)

    # ... rest of the OpenAIHandler implementation from your reference ...

def get_fn(model_name: str, preprocess: Callable, postprocess: Callable, api_key: str):
    def fn(message, history):
        inputs = preprocess(message, history)
        client = OpenAI(api_key=api_key)
        completion = client.chat.completions.create(
            model=model_name,
            messages=inputs["messages"],
            stream=True,
        )
        response_text = ""
        for chunk in completion:
            delta = chunk.choices[0].delta.content or ""
            response_text += delta
            yield postprocess(response_text)

    return fn


def get_image_base64(url: str, ext: str):
    with open(url, "rb") as image_file:
        encoded_string = base64.b64encode(image_file.read()).decode('utf-8')
    return "data:image/" + ext + ";base64," + encoded_string


def handle_user_msg(message: str):
    if type(message) is str:
        return message
    elif type(message) is dict:
        if message["files"] is not None and len(message["files"]) > 0:
            ext = os.path.splitext(message["files"][-1])[1].strip(".")
            if ext.lower() in ["png", "jpg", "jpeg", "gif", "pdf"]:
                encoded_str = get_image_base64(message["files"][-1], ext)
            else:
                raise NotImplementedError(f"Not supported file type {ext}")
            content = [
                    {"type": "text", "text": message["text"]},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": encoded_str,
                        }
                    },
                ]
        else:
            content = message["text"]
        return content
    else:
        raise NotImplementedError


def get_interface_args(pipeline):
    if pipeline == "chat":
        inputs = None
        outputs = None

        def preprocess(message, history):
            messages = []
            files = None
            for user_msg, assistant_msg in history:
                if assistant_msg is not None:
                    messages.append({"role": "user", "content": handle_user_msg(user_msg)})
                    messages.append({"role": "assistant", "content": assistant_msg})
                else:
                    files = user_msg
            if type(message) is str and files is not None:
                message = {"text":message, "files":files}
            elif type(message) is dict and files is not None:
                if message["files"] is None or len(message["files"]) == 0:
                    message["files"] = files
            messages.append({"role": "user", "content": handle_user_msg(message)})
            return {"messages": messages}

        postprocess = lambda x: x
    else:
        # Add other pipeline types when they will be needed
        raise ValueError(f"Unsupported pipeline type: {pipeline}")
    return inputs, outputs, preprocess, postprocess


def get_pipeline(model_name, enable_voice=False):
    if enable_voice or "realtime" in model_name.lower():
        return "voice"
    return "chat"


def registry(name: str, token: str | None = None, enable_voice: bool = False, twilio_sid: str | None = None, twilio_token: str | None = None, **kwargs):
    """
    Create a Gradio Interface for a model on OpenAI.

    Parameters:
        - name (str): The name of the OpenAI model.
        - token (str, optional): The API key for OpenAI.
        - enable_voice (bool, optional): Force enable voice interface regardless of model name.
        - twilio_sid (str, optional): Twilio Account SID for TURN server.
        - twilio_token (str, optional): Twilio Auth Token for TURN server.
    """
    api_key = token or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    pipeline = get_pipeline(name, enable_voice)
    
    if pipeline == "voice":
        with gr.Blocks() as interface:
            with gr.Row() as api_key_row:
                api_key_input = gr.Textbox(
                    label="OpenAI API Key",
                    placeholder="Enter your OpenAI API Key",
                    value=api_key,
                    type="password",
                    visible=False
                )
            with gr.Row() as row:
                webrtc = WebRTC(
                    label="Conversation",
                    modality="audio",
                    mode="send-receive",
                    rtc_configuration=get_twilio_turn_credentials(twilio_sid, twilio_token),
                )
                
            webrtc.stream(
                OpenAIHandler(),
                inputs=[webrtc, api_key_input],
                outputs=[webrtc],
                time_limit=90,
                concurrency_limit=10,
            )
    else:
        # Existing chat interface code
        inputs, outputs, preprocess, postprocess = get_interface_args(pipeline)
        fn = get_fn(name, preprocess, postprocess, api_key)
        
        if pipeline == "chat":
            interface = gr.ChatInterface(fn=fn, multimodal=True, **kwargs)
        else:
            interface = gr.Interface(fn=fn, inputs=inputs, outputs=outputs, **kwargs)

    return interface

def update_chatbot(chatbot: list[dict], response):
    chatbot.append({"role": "assistant", "content": response.transcript})
    return chatbot
