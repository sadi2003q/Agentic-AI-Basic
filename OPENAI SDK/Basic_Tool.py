from agents import Agent, WebSearchTool, Runner
from QuickStart import gemini_model
import asyncio
from dotenv import load_dotenv

load_dotenv()


agent = Agent(
    name="Assistant",
    instructions="You are a Powerful web service",
    model=gemini_model,
    tools=[WebSearchTool()]
)


async def main():
    result = await Runner.run(agent, "Which shop is the best shop for burger right now in Dhaka")
    print(result.final_output)


if __name__ == '__main__':
    asyncio.run(main())