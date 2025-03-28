from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
load_dotenv()


@CrewBase
class SummaryGenerator():
    """SummaryGenerator crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def summary_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['summary_agent'],
            verbose=True
        )

    @task
    def summary_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['summary_agent_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the SummaryGenerator crew"""

        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,

        )
