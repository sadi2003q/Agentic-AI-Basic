from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from Gmail_Tool.gmail_tools import gmail_tool


@CrewBase
class Gmail_Crew:
    """Gmail Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def gmail_draft_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["gmail_draft_agent"],
            tools=[gmail_tool()]

        )

    @task
    def gmail_draft_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config["gmail_draft_agent_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
        )


if __name__ == "__main__":
    gmail_tool.func()