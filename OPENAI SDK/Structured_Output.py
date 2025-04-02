from agents import Agent, Runner
from QuickStart import gemini_model
from pydantic import BaseModel
from dotenv import load_dotenv
import asyncio

load_dotenv()


class Email_Structure(BaseModel):
    subject: str
    body: str


email_agent = Agent(
    name="email agent",
    model=gemini_model,
    output_type=Email_Structure
)


async def main():

    input_data = "Write an email to John confirming our meeting at 3 PM tomorrow."

    result = await Runner.run(email_agent, input_data)

    print(f"Subject: {result.final_output.subject}")
    print(f"Body: {result.final_output.body}")


if __name__ == "__main__":
    asyncio.run(main())
