#!/usr/bin/env python
from io import BytesIO
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from dotenv import load_dotenv
from pydub.utils import make_chunks
from openai import OpenAI
from pydub import AudioSegment
from meeting_minutes_crew import Meeting_Minute_crew

load_dotenv()
client = OpenAI()


class Meeting_State(BaseModel):
    transcript: str = ""
    meeting_minute: str = ""


class Meeting_Flow(Flow[Meeting_State]):

    @start()
    def transcribe_meeting(self):
        path = '/Users/sadi_/Coding/AI Agents/Meeting Partner/audio.wav'

        # Load the audio file
        audio = AudioSegment.from_wav(path)

        # Define chunk length in milliseconds (e.g., 1 minute = 60,000 ms)
        chunk_length_ms = 60000
        chunks = make_chunks(audio, chunk_length_ms)

        # Transcribe each chunk
        full_transcription = ""
        self.state.transcript = full_transcription
        print(f"Transcription: {self.state.transcript}")
        for i, chunk in enumerate(chunks):
            if i > 1:
                break
            print(f"Transcribing chunk {i + 1}/{len(chunks)}")
            chunk_path = f"chunk_{i}.wav"
            chunk.export(chunk_path, format="wav")

            # Transcribe with Whisper
            with open(chunk_path, "rb") as audio_file:
                transcript = client.audio.transcriptions.create(
                    model="whisper-1", file=audio_file
                )
                full_transcription += transcript.text + "\n"

        self.state.transcript = full_transcription
        print(f"Transcription: {self.state.transcript}")

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
    """
    this will make an animated plot of the crew by making an
    new HTML file in the directory
    :return:
    """
    flow = Meeting_Flow()
    flow.plot()


if __name__ == "__main__":
    kickoff()
