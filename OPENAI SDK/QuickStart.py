from openai import OpenAI, AsyncOpenAI
from agents import Agent, Runner, OpenAIChatCompletionsModel
from dotenv import load_dotenv

load_dotenv()

client = AsyncOpenAI(
    api_key="AIzaSyAJc94qpGwtMfboytY4Tv1Wn5UBbBj0jiA",
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

gemini_model = OpenAIChatCompletionsModel(
    model="gemini-2.0-flash",
    openai_client=client
)

gemini_agent = Agent(
    name="Gemini Agent",
    instructions="You are a helpful assistant powered by Gemini.",
    model=gemini_model
)


async def main():
    result = await Runner.run(gemini_agent, input="Explain how AI works in simple terms.")
    print(result.final_output)

