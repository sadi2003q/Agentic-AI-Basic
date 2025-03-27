from crewai.flow.flow import Flow, listen, start
from pydantic import BaseModel
from poem_generator.src.poem_generator.crew import PoemGenerator
from summary_generator.src.summary_generator.crew import SummaryGenerator


class combine_model(BaseModel):
    poem: str = ""
    summary: str = ""


class Poem(Flow[combine_model]):

    def __init__(self, topic: str = "AI"):
        super().__init__()
        self.topic = topic

    @start()
    def GeneratePoem(self):
        inputs = {
            'topic': self.topic
        }
        try:
            self.state.poem = PoemGenerator().crew().kickoff(inputs=inputs)
        except Exception as e:
            raise Exception(f"An error occurred while running the crew: {e}")

    @listen(GeneratePoem)
    def summarise_poem(self):
        inputs = {
            'passage': str(self.state.poem)
        }
        try:
            self.state.summary = SummaryGenerator().crew().kickoff(inputs=inputs)
            return self.state.summary, self.state.poem
        except Exception as e:
            raise Exception(f"An error occurred while running the crew: {e}")


if __name__ == "__main__":
    poem = Poem(topic="""
    write me a poem about the July Protest of 2024 in bangladesh and how bangladesh got new independence from Shekh Hasina
    """)
    summary, poem = poem.kickoff()

    print("Here is the POEM")
    print(poem)

    print("Here is the Summary")
    print(summary)