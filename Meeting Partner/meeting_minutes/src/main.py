#!/usr/bin/env python
import google.generativeai as genai
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from dotenv import load_dotenv
from pydub.utils import make_chunks
from pydub import AudioSegment
import os
import Summarisation
from Email_Agent import Email_Agent

GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
load_dotenv()


class Meeting_State(BaseModel):
    transcript: str = ""
    meeting_minute: str = ""


class Meeting_Flow(Flow[Meeting_State]):

    @start()
    def transcribe_meeting(self):
        path = '/Users/sadi_/Coding/AI Agents/Meeting Partner/meeting_minutes/src/audio.wav'
        audio = AudioSegment.from_wav(path)

        chunk_length_ms = 60000
        chunks = make_chunks(audio, chunk_length_ms)

        for i, chunk in enumerate(chunks):
            if i > 2:
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
            finally:
                if os.path.exists(chunk_path):
                    os.remove(chunk_path)

    @listen(transcribe_meeting)
    def make_summary(self):
        input = self.state.transcript
        print(type(input))
        summary = Summarisation.Summarise_Info(text=input)
        Summarisation.Save_Markdown_file(summary)

    @listen(make_summary)
    def sent_email(self):
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


if __name__ == "__main__":
    kickoff()
