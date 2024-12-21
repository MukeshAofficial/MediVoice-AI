import asyncio
import base64
import io
import os
import sys
import traceback

import cv2
import pyaudio
import PIL.Image

from google import genai
from dotenv import load_dotenv

load_dotenv()




GOOGLE_API_KEY = "YOUR-API-KEY"
if not GOOGLE_API_KEY:
    raise ValueError("API_KEY environment variable not set. Check your .env file.")

if sys.version_info < (3, 11, 0):
    import taskgroup, exceptiongroup
    asyncio.TaskGroup = taskgroup.TaskGroup
    asyncio.ExceptionGroup = exceptiongroup.ExceptionGroup

FORMAT = pyaudio.paInt16
CHANNELS = 1
SEND_SAMPLE_RATE = 16000
RECEIVE_SAMPLE_RATE = 24000
CHUNK_SIZE = 512

MODEL = "models/gemini-2.0-flash-exp"

client = genai.Client(http_options={'api_version': 'v1alpha'}, api_key=GOOGLE_API_KEY)

INITIAL_PROMPT = """You are a helpful and friendly virtual assistant named "Gemini Assist." 
Respond to user questions in a concise and informative manner. If you don't know the answer, 
say "I'm sorry, I don't have that information." Be polite and professional in your responses.
"""
CONFIG = {"generation_config": {"response_modalities": ["AUDIO"]}}

pya = pyaudio.PyAudio()

class AudioLoop:
    def __init__(self):
        self.audio_in_queue = asyncio.Queue()
        self.audio_out_queue = asyncio.Queue()
        self.video_out_queue = asyncio.Queue()
        self.session = None
        self.conversation_history = []  # Store conversation history

    async def send_text(self):
        while True:
            text = await asyncio.to_thread(input, "You: ")
            if text.lower() == "q":
                break

            self.conversation_history.append({"role": "user", "parts": [{"text": text}]})
            await self.send_to_gemini()


    async def send_to_gemini(self):
        try:
            response = client.generate_text(
                model=MODEL,
                prompt=self.conversation_history,
                generation_config=CONFIG,
            )
            if response.result:
                print("Gemini Assist:", response.result)
                self.conversation_history.append({"role": "assistant", "parts": [{"text": response.result}]})
        except Exception as e:
            print(f"Error communicating with Gemini: {e}")

    def _get_frame(self, cap):
        ret, frame = cap.read()
        if not ret:
            return None
        img = PIL.Image.fromarray(frame)
        img.thumbnail([1024, 1024])
        image_io = io.BytesIO()
        img.save(image_io, format="jpeg")
        image_io.seek(0)
        mime_type = "image/jpeg"
        image_bytes = image_io.read()
        return {"mime_type": mime_type, "data": base64.b64encode(image_bytes).decode()}

    async def get_frames(self):
        cap = await asyncio.to_thread(cv2.VideoCapture, 0)
        while True:
            frame = await asyncio.to_thread(self._get_frame, cap)
            if frame is None:
                break
            await asyncio.sleep(1.0)
            self.video_out_queue.put_nowait(frame)
        cap.release()

    async def send_frames(self):
        while True:
            frame = await self.video_out_queue.get()
            await self.session.send(frame)

    async def listen_audio(self):
        mic_info = pya.get_default_input_device_info()
        stream = await asyncio.to_thread(
            pya.open, format=FORMAT, channels=CHANNELS, rate=SEND_SAMPLE_RATE,
            input=True, input_device_index=mic_info["index"], frames_per_buffer=CHUNK_SIZE
        )
        while True:
            data = await asyncio.to_thread(stream.read, CHUNK_SIZE)
            self.audio_out_queue.put_nowait(data)

    async def send_audio(self):
        while True:
            chunk = await self.audio_out_queue.get()
            await self.session.send({"data": chunk, "mime_type": "audio/pcm"})

    async def receive_audio(self):
        while True:
            async for response in self.session.receive():
                server_content = response.server_content
                if server_content and server_content.model_turn:
                    for part in server_content.model_turn.parts:
                        if part.text:
                            print(part.text, end="")
                            self.conversation_history.append({"role": "assistant", "parts": [{"text": part.text}]}) # Add assistant's text response to history
                        elif part.inline_data:
                            self.audio_in_queue.put_nowait(part.inline_data.data)
                if server_content and server_content.turn_complete:
                    print("Turn complete")
                    while not self.audio_in_queue.empty():
                        self.audio_in_queue.get_nowait()

    async def play_audio(self):
        stream = await asyncio.to_thread(
            pya.open, format=FORMAT, channels=CHANNELS, rate=RECEIVE_SAMPLE_RATE, output=True
        )
        while True:
            bytestream = await self.audio_in_queue.get()
            await asyncio.to_thread(stream.write, bytestream)

    async def run(self):
        self.conversation_history.append({"role": "user", "parts": [{"text": INITIAL_PROMPT}]}) # Send initial prompt
        async with (
            client.aio.live.connect(model=MODEL, config=CONFIG) as session
        ):
            self.session = session
            tasks = [
                self.send_text(),
                self.listen_audio(),
                self.send_audio(),
                self.get_frames(),  # If you want to enable video input
                self.send_frames(), # If you want to enable video input
                self.receive_audio(),
                self.play_audio(),
            ]
            await asyncio.gather(*tasks)


if __name__ == "__main__":
    main = AudioLoop()
    asyncio.run(main.run())