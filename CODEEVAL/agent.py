from google.adk.agents import Agent
from google.adk.tools.agent_tool import AgentTool
from .sub_agents.code_executor.agent import code_executor

from .tools.tools import get_current_time

root_agent = Agent(
    name="manager",
    model="gemini-2.0-flash",
    description="Manager agent",
    instruction="""
    You are an interviewer agent that is responsible for managing the coding interview section. Provide a positive remark if the code is correct. If the code is incorrect, provide ONE small hint to nudge the user. Do not reveal the complete solution.
    Write appropriate test cases to the question and use the tools available to you to execute the code to verify if the code is correct.
    You have access to the following tools:
    - code_executor: Executes the code and provides feedback on it.
    """,
    tools=[
        AgentTool(code_executor),
        get_current_time,
    ],
)

