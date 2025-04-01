from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from dotenv import load_dotenv
from gtts import gTTS

load_dotenv()


class TextToSpeech:
    def __init__(self, topic: str):
        self.model = ChatOpenAI(model='gpt-4o-mini')
        self.topic = topic
        self.prompt = PromptTemplate(
            template="tell me a short simple report about {topic}",
            input_variables=['topic']
        )

    def generate(self):
        chain = self.prompt | self.model | StrOutputParser()
        response = chain.invoke({'topic': self.topic})
        print(response)
        tts = gTTS(text=response, lang="en")
        tts.save("report.mp3")


if __name__ == '__main__':
    llm = TextToSpeech("Cold War")
    llm.generate()
