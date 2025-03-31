#!/usr/bin/env python
from io import BytesIO
from pydantic import BaseModel
from crewai.flow import Flow, listen, start
from dotenv import load_dotenv
from pydub.utils import make_chunks
from openai import OpenAI
from pydub import AudioSegment
<<<<<<< HEAD:Meeting Partner/meeting_minutes/src/meeting_minutes/main.py
from meeting_minutes_crew import Meeting_Minute_crew
=======
import os
import Summarisation
from Email_Agent import Email_Agent
>>>>>>> test:Meeting Partner/meeting_minutes/src/main.py

load_dotenv()
client = OpenAI()


class Meeting_State(BaseModel):
    transcript: str = ""
    meeting_minute: str = ""


class Meeting_Flow(Flow[Meeting_State]):

    @start()
    def transcribe_meeting(self):
<<<<<<< HEAD:Meeting Partner/meeting_minutes/src/meeting_minutes/main.py
        path = '/Users/sadi_/Coding/AI Agents/Meeting Partner/audio.wav'

        # Load the audio file
=======
        path = 'audio.wav'
>>>>>>> test:Meeting Partner/meeting_minutes/src/main.py
        audio = AudioSegment.from_wav(path)

        # Define chunk length in milliseconds (e.g., 1 minute = 60,000 ms)
        chunk_length_ms = 60000
        chunks = make_chunks(audio, chunk_length_ms)

        # Transcribe each chunk
        full_transcription = ""
        self.state.transcript = full_transcription
        print(f"Transcription: {self.state.transcript}")
        for i, chunk in enumerate(chunks):
<<<<<<< HEAD:Meeting Partner/meeting_minutes/src/meeting_minutes/main.py
            if i > 1:
=======
            if i > 2:
>>>>>>> test:Meeting Partner/meeting_minutes/src/main.py
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

<<<<<<< HEAD:Meeting Partner/meeting_minutes/src/meeting_minutes/main.py
        self.state.transcript = full_transcription
        print(f"Transcription: {self.state.transcript}")
=======
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
>>>>>>> test:Meeting Partner/meeting_minutes/src/main.py

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


<<<<<<< HEAD:Meeting Partner/meeting_minutes/src/meeting_minutes/main.py
def plot():
    """
    this will make an animated plot of the crew by making an
    new HTML file in the directory
    :return:
    """
    flow = Meeting_Flow()
    flow.plot()


=======
>>>>>>> test:Meeting Partner/meeting_minutes/src/main.py
if __name__ == "__main__":
    kickoff()
