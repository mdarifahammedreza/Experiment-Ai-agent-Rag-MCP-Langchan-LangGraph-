from langgraph.prebuilt import create_react_agent
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import Tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    SystemMessage,
)
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv
import os
load_dotenv()
google_api_key = os.environ.get("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY is not set in your environment!")
model = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash", 
    google_api_key=google_api_key
)

agent = create_react_agent(
    model=model,
    tools=[get_weather_tool],
)

class AgentState(TypedDict):
    message: Annotated[List[Union[HumanMessage,AIMessage]]]

#define tool
@tool
def add(a: int, b: int) -> int:
    """Add two numbers."""
    return a + b

tools = [add]