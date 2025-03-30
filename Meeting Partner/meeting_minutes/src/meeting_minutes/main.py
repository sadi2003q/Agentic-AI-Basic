#!/usr/bin/env python
import google.generativeai as genai
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from dotenv import load_dotenv
from pydub.utils import make_chunks
from pydub import AudioSegment
from meeting_minutes_crew import Meeting_Minute_crew
import os

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
load_dotenv()


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

        for i, chunk in enumerate(chunks):
            if i > 5:
                break
            print(f"Transcribing chunk {i + 1}/{len(chunks)}")
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")

            try:
                model = genai.GenerativeModel("gemini-1.5-pro")

                with open(chunk_path, "rb") as audio_file:
                    audio_data = audio_file.read()

                content = {
                    "parts": [
                        {
                            "mime_type": "audio/wav",
                            "data": audio_data,
                        },
                        "Transcribe this audio to text.",
                    ],
                }

                response = model.generate_content(content)
                self.state.transcript += " " + response.text

            except Exception as e:
                return f"An error occurred: {e}"
        print(self.state.transcript)

    @listen(transcribe_meeting)
    def generate_transcription(self):
        print("Generating Meeting Minutes")

        crew = Meeting_Minute_crew()

        inputs = {
            "transcript": self.state.transcript
        }

        meeting_minutes = crew.crew().kickoff(inputs)
        self.state.meeting_minute = meeting_minutes


def kickoff():
    flow = Meeting_Flow()
    flow.kickoff()


def plot():
    flow = Meeting_Flow()
    flow.plot()


if __name__ == "__main__":
    Meeting_Flow().kickoff()
    plot()
