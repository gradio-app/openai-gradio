import os
import asyncio
import base64
import time
from threading import Event, Thread

import gradio as gr
import numpy as np
from openai import OpenAI
from gradio_webrtc import (
    AdditionalOutputs,
    StreamHandler,
    WebRTC,
    get_twilio_turn_credentials,
)
from pydub import AudioSegment

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

class RealtimeHandler(StreamHandler):
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
        # Initialize Event objects first
        self.args_set = Event()
        self.quit = Event()
        self.connected = Event()
        self.reset_state()

    def reset_state(self):
        """Reset connection state for new recording session"""
        self.connection = None
        self.args_set.clear()
        self.quit.clear()
        self.connected.clear()
        self.thread = None
        self._generator = None
        self.current_session = None

    def copy(self):
        return RealtimeHandler(
            expected_layout=self.expected_layout,
            output_sample_rate=self.output_sample_rate,
            output_frame_size=self.output_frame_size,
        )

    def _initialize_connection(self, api_key: str):
        self.client = OpenAI(api_key=api_key)
        with self.client.beta.realtime.connect(
            model="gpt-4o-realtime-preview-2024-10-01"
        ) as conn:
            conn.session.update(session={"turn_detection": {"type": "server_vad"}})
            self.connection = conn
            self.connected.set()
            self.current_session = conn.session
            while not self.quit.is_set():
                time.sleep(0.25)

    async def fetch_args(self):
        if self.channel:
            self.channel.send("tick")

    def set_args(self, args):
        super().set_args(args)
        self.args_set.set()

    def receive(self, frame: tuple[int, np.ndarray]) -> None:
        if not self.channel:
            return
        try:
            # Initialize connection if needed
            if not self.connection:
                asyncio.run_coroutine_threadsafe(self.fetch_args(), self.loop)
                self.args_set.wait()
                self.thread = Thread(
                    target=self._initialize_connection, args=(self.latest_args[-1],)
                )
                self.thread.start()
                self.connected.wait()
            
            # Send audio data
            assert self.connection, "Connection not initialized"
            sample_rate, array = frame
            array = array.squeeze()
            audio_message = encode_audio(sample_rate, array)
            
            # Send the audio data
            self.connection.input_audio_buffer.append(audio=audio_message)
            
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
            self.quit.set()
            self.connection.close()
            if self.thread:
                self.thread.join(timeout=5)
            self.reset_state()  # Reset state after shutdown

def update_chatbot(chatbot: list[dict], response):
    chatbot.append({"role": "assistant", "content": response.transcript})
    return chatbot

def registry(name: str, token: str | None = None, twilio_sid: str | None = None, twilio_token: str | None = None, **kwargs):
    api_key = token or os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY environment variable is not set.")

    # Use default Twilio credentials if none provided
    twilio_sid = twilio_sid or os.environ.get("TWILIO_ACCOUNT_SID")
    twilio_token = twilio_token or os.environ.get("TWILIO_AUTH_TOKEN")

    with gr.Blocks() as interface:
        # Set initial visibility based on whether API key is provided
        show_api_input = api_key is None
        
        with gr.Row(visible=show_api_input) as api_key_row:
            api_key_input = gr.Textbox(
                label="OpenAI API Key",
                placeholder="Enter your OpenAI API Key",
                value=api_key,
                type="password",
            )
            
        with gr.Row(visible=not show_api_input) as row:
            webrtc = WebRTC(
                label="Conversation",
                modality="audio",
                mode="send-receive",
                rtc_configuration=get_twilio_turn_credentials(twilio_sid, twilio_token),
            )
                
            webrtc.stream(
                RealtimeHandler(),
                inputs=[webrtc, api_key_input],
                outputs=[webrtc],
                time_limit=90,
                concurrency_limit=2,
            )
            
        api_key_input.submit(
            lambda: (gr.update(visible=False), gr.update(visible=True)),
            None,
            [api_key_row, row],
        )

    return interface