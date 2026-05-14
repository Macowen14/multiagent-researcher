from typing import Annotated, TypedDict
from langgraph.graph.message import add_messages

class MultiAgentState(TypedDict):
    """
    Custom state for the Multi-Agent System.
    Separates conversational chat history from large payload data to prevent context window issues.
    """
    messages: Annotated[list, add_messages]
    research_report: str      # Store the raw report here
    scraped_data: str         # Store heavy tool outputs here
    document_draft: str       # Store the Writer's draft here
    approval_status: str      # "PENDING" or "APPROVED"
