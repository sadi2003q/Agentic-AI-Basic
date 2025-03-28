import sys
import warnings
import io
from pydub import AudioSegment
from crew import Meeting_Minutes_crew
import google.generativeai as genai
from dotenv import load_dotenv
from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel

load_dotenv()
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


class transcript_generator(BaseModel):
    """
    class for Generating the Summary of the Meeting
    """

    transcript: str = ''
    meeting_minute: str = ''


class transcript_generator_flow(Flow[transcript_generator]):

    @start()
    def generate_audio(self, path='/Users/sadi_/Coding/AI Agents/Meeting Partner/audio.wav'):

        try:
            model = genai.GenerativeModel("gemini-2.0-flash")

            audio = AudioSegment.from_wav(path)

            audio_2min = audio[:120000]

            # Convert the extracted portion into bytes
            audio_buffer = io.BytesIO()
            audio_2min.export(audio_buffer, format="wav")
            audio_data = audio_buffer.getvalue()

            content = [
                {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": audio_data
                    }
                },
                {
                    "text": "Transcribe the audio to text"
                }
            ]

            self.state.transcript = model.generate_content(content)
            self.state.generating_condition = True
        except Exception as e:
            print(f"error : {e}")
            self.state.generating_condition = True
            sys.exit(1)

    @listen(generate_audio)
    def generate_summary(self):
        input_text = {
            'transcript': str(self.state.transcript)
        }

        try:
            if self.state.generating_condition:
                Meeting_Minutes_crew.crew.kick_off_meeting(input=self.state.meeting_minute)
        except Exception as e:
            print(f"error : {e}")
            self.state.generating_condition = False
            sys.exit(1)


if __name__ == "__main__":
    flow = transcript_generator_flow()
    flow.kickoff()
