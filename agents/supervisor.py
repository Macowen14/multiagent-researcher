import enum
import logging
from typing import List, Callable
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage
from langgraph.graph import MessagesState
from langgraph.types import Command
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)

def create_supervisor(
    llm: BaseChatModel,
    members: List[str],
    system_prompt: str
) -> Callable[[MessagesState], Command]:
    """
    Creates a highly reusable supervisor node for LangGraph.
    
    Args:
        llm: An initialized BaseChatModel.
        members: A list of worker agent names that the supervisor can route to.
        system_prompt: The instruction prompt for the supervisor. Use {members}
                       to format the member list inside the prompt.
                       
    Returns:
        A LangGraph node function that returns a Command to route execution.
    """
    options = members + ["__end__"]
    
    # We create a dynamic Enum representing the exact strings we want to allow.
    # This ensures the LLM is physically constrained to pick a valid route.
    OptionsEnum = enum.Enum("OptionsEnum", {opt: opt for opt in options})
    
    class Route(BaseModel):
        next: OptionsEnum = Field(description="The name of the next worker to route to, or __end__ if finished.")
        
    def supervisor_node(state: MessagesState) -> Command:
        """The actual node function that runs in LangGraph."""
        logger.info(f"Supervisor evaluating state with {len(state['messages'])} messages...")
        
        # Format the system prompt with the list of members
        formatted_prompt = system_prompt.format(members=", ".join(members))
        messages = [SystemMessage(content=formatted_prompt)] + state["messages"]
        
        # Force the LLM to output our exact Route schema
        response = llm.with_structured_output(Route).invoke(messages)
        
        next_node = response.next.value
        logger.info(f"Supervisor decision: Routing to -> {next_node}")
        
        return Command(goto=next_node)
        
    return supervisor_node
