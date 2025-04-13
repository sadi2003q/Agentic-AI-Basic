from agents import Agent, FileSearchTool, Runner, WebSearchTool
from dotenv import load_dotenv
import asyncio

load_dotenv()



agent = Agent(
    name="Assistant",
    tools=[
        WebSearchTool(),
        FileSearchTool(
            max_num_results=3,
            vector_store_ids=["VECTOR_STORE_ID"],
        ),
    ],
)


async def main():
    result = await Runner.run(agent, "Which coffee shop should I go to, taking into account my preferences and the weather today in SF?")
    print(result.final_output)


if __name__ == '__main__':
    asyncio.run(main())