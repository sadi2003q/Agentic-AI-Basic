from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileWriterTool

file_writer_tool_summary = FileWriterTool(file_name='summary.md')
file_writer_tool_sentiment = FileWriterTool(file_name='sentiment.md')
file_writer_tool_action_item = FileWriterTool(file_name='action_item.md')


@CrewBase
class Meeting_Minute_crew:
    """Poem Crew"""

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

    @agent
    def meeting_minute_summariser(self) -> Agent:
        return Agent(
            config=self.agents_config["meeting_minute_summariser"],
            tools=[file_writer_tool_summary, file_writer_tool_action_item, file_writer_tool_sentiment],
        )

    @agent
    def meeting_minute_writer(self) -> Agent:
        return Agent(
            config=self.agents_config["meeting_minute_writer"],
        )

    @task
    def meeting_minute_summariser_task(self) -> Task:
        return Task(
            config=self.tasks_config["meeting_minute_summariser_task"],
        )

    @task
    def meeting_minute_writer_task(self) -> Task:
        return Task(
            config=self.tasks_config["meeting_minute_writer_task"],
        )

    @crew
    def crew(self) -> Crew:
        """Creates the Research Crew"""

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
