
from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai_tools import FileWriterTool

file_writer_tool_summary = FileWriterTool(file_name="summary.md", directory='/Users/sadi_/Coding/AI Agents/Meeting Partner')
file_writer_tool_action_item = FileWriterTool(file_name="action_item.md", directory='/Users/sadi_/Coding/AI Agents/Meeting Partner')
file_writer_tool_task = FileWriterTool(file_name="sentiment.md", directory='/Users/sadi_/Coding/AI Agents/Meeting Partner')


@CrewBase
class Meeting_Minutes_crew():

    """AudioTranscriptGenerator crew"""

    agents_config = 'config/agents.yaml'
    tasks_config = 'config/tasks.yaml'

    @agent
    def meeting_minutes_summariser(self) -> Agent:
        return Agent(
            config=self.agents_config['meeting_minute_summariser'],
            tools=[file_writer_tool_task, file_writer_tool_summary, file_writer_tool_action_item],
            verbose=True
        )

    @agent
    def meeting_minute_writer(self) -> Agent:
        return Agent(
            config=self.agents_config['meeting_minute_writer'],
            verbose=True
        )

    @task
    def meeting_minutes_summariser_task(self) -> Task:
        return Task(
            config=self.tasks_config['meeting_minutes_summary_task'],
        )

    @task
    def meeting_minute_writer_task(self) -> Task:
        return Task(
            config=self.tasks_config['meeting_minutes_writing_task'],
            output_file='summary.md'
        )

    @crew
    def crew(self) -> Crew:
        """Creates the AudioTranscriptGenerator crew"""
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,

        )
