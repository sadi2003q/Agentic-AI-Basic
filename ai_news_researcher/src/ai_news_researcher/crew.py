
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import SerperDevTool, ScrapeWebsiteTool, FileWriterTool
from dotenv import load_dotenv
load_dotenv()

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class AiNewsResearcher():
    """AiNewsResearcher crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def retrieve_news(self) -> Agent:
        return Agent(
            config=self.agents_config['retrieve_news'],
            tool=[SerperDevTool()],
            verbose=True
        )

    @agent
    def website_scraper(self) -> Agent:
        return Agent(
            config=self.agents_config['website_scraper'],
            tool=[ScrapeWebsiteTool()],
            verbose=True
        )

    @agent
    def ai_news_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['ai_news_writer'],
            tool=[],
            verbose=True
        )

    @agent
    def file_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['file_writer'],
            tool=[FileWriterTool()],
            verbose=True
        )

    @task
    def retrieve_news_task(self) -> Task:
        return Task(
            config=self.tasks_config['retrieve_news_task'],
        )

    @task
    def website_scrape_task(self) -> Task:
        return Task(
            config=self.tasks_config['website_scrape_task'],
        )

    @task
    def ai_news_write_task(self) -> Task:
        return Task(
            config=self.tasks_config['ai_news_write_task'],
        )

    @task
    def file_write_task(self) -> Task:
        return Task(
            config=self.tasks_config['file_write_task'],
            output_file='report.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AiNewsResearcher crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,

        )
