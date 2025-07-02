from google.adk.agents import Agent
from google.adk.tools import google_search
from google.adk.tools import built_in_code_execution

code_executor = Agent(
    name="code_executor",
    model="gemini-2.0-flash",
    description="Code Executor agent",
    instruction="""
    You are an agent that is responsible for executing code and providing feedback on it.
    You have access to the following tools:
    - built_in_code_execution
    """,
    tools=[built_in_code_execution],
)
