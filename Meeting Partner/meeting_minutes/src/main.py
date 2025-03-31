#!/usr/bin/env python
import os
from io import BytesIO
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from dotenv import load_dotenv
from pydub.utils import make_chunks
from openai import OpenAI
from pydub import AudioSegment
import Summarisation
from Email_Agent import Email_Agent


load_dotenv()
client = OpenAI()


class Meeting_State(BaseModel):
    transcript: str = ""
    meeting_minute: str = ""


class Meeting_Flow(Flow[Meeting_State]):
    @start()
    def transcribe_meeting(self):
        path = '/Users/sadi_/Coding/AI Agents/Meeting Partner/audio.wav'
        audio = AudioSegment.from_wav(path)

        chunk_length_ms = 60000
        chunks = make_chunks(audio, chunk_length_ms)

        full_transcription = ""
        self.state.transcript = full_transcription
        print(f"Transcription: {self.state.transcript}")

        for i, chunk in enumerate(chunks):
            print(f"Transcribing chunk {i + 1}/{len(chunks)}")
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")

            try:
                with open(chunk_path, "rb") as audio_file:
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1", file=audio_file
                    )
                    full_transcription += transcript.text + "\n"

                    content = {
                        "parts": [
                            {"mime_type": "audio/wav", "data": audio_file.read()},
                            "Transcribe this audio to text.",
                        ],
                    }

                    response = client.generate_content(content)
                    self.state.transcript += " " + response.text

            except Exception as e:
                print(f"An error occurred: {e}")

            finally:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)

    @listen(transcribe_meeting)
    def make_summary(self):
        input_text = self.state.transcript
        summary = Summarisation.Summarise_Info(text=input_text)
        Summarisation.Save_Markdown_file(summary)

    @listen(make_summary)
    def send_email(self):
        agent = Email_Agent(
            'adnanabdullah625@gmail.com',
            'adnan.sadi@northsouth.edu',
            '/Users/sadi_/Coding/AI Agents/Meeting Partner/Files/markdown_file.md',
            str(self.state.transcript)
        )
        agent.Proceed_Email()


def kickoff():
    flow = Meeting_Flow()
    flow.kickoff()


def plot():
    flow = Meeting_Flow()
    flow.plot()


if __name__ == "__main__":
    kickoff()
