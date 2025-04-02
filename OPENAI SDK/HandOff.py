from agents import Agent, Runner
from QuickStart import gemini_model
from dotenv import load_dotenv
import asyncio

load_dotenv()

history_tutor_agent = Agent (
    name="History tutor Agent",
    handoff_description="You are an Expert Historian Agent",
    instructions="Provide user accurate information about History",
    model=gemini_model
)

Math_tutor_agent = Agent (
    name="Math tutor Agent",
    handoff_description="You are an expert Mathematician",
    instructions="Provide correct Math solution",
    model=gemini_model
)

final_agent = Agent(
    name="Final agent",
    instructions="Determine which agent to use based on the query",
    handoffs=[Math_tutor_agent, history_tutor_agent],
    model=gemini_model
)


async def main():
    result = await Runner.run(final_agent, 'Tell me the history of France Independence')
    print(result.final_output)


if __name__ == "__main__":
    asyncio.run(main())