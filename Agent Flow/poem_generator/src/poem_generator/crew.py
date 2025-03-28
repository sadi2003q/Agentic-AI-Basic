from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from dotenv import load_dotenv
load_dotenv()


@CrewBase
class PoemGenerator():

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def poem_agent(self) -> Agent:
        return Agent(
            config=self.agents_config['Poem_Generator'],
            verbose=True
        )

    @task
    def poem_agent_task(self) -> Task:
        return Task(
            config=self.tasks_config['poem_agent_task'],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the PoemGenerator crew"""


        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,

        )
