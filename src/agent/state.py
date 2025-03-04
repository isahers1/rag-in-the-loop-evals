"""Define the state structures for the agent."""

from typing_extensions import TypedDict, Annotated
from langgraph.graph.message import add_messages


class State(TypedDict):
    messages: Annotated[list, add_messages]
    queries: list[str]
    question: str
    context: str
