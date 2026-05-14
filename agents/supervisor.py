import enum
import logging
from typing import List, Callable
from pydantic import BaseModel, Field
from langchain_core.messages import SystemMessage, HumanMessage
from langgraph.types import Command
from langchain_core.language_models.chat_models import BaseChatModel

from models.state import MultiAgentState

logger = logging.getLogger(__name__)


def create_supervisor(
    llm: BaseChatModel,
    members: List[str],
    system_prompt: str,
    max_consecutive_same: int = 2,
) -> Callable[[MultiAgentState], Command]:
    """
    Creates a highly reusable supervisor node for LangGraph.

    Args:
        llm: An initialized BaseChatModel.
        members: A list of worker agent names that the supervisor can route to.
        system_prompt: The instruction prompt for the supervisor. Use {members}
                       to format the member list inside the prompt.
        max_consecutive_same: Safety limit — if the same worker is routed to
                              this many times in a row, force __end__.

    Returns:
        A LangGraph node function that returns a Command to route execution.
    """
    options = members + ["__end__"]

    # Dynamic Enum ensures the LLM is physically constrained to pick a valid route.
    OptionsEnum = enum.Enum("OptionsEnum", {opt: opt for opt in options})

    class Route(BaseModel):
        reason: str = Field(
            description="A brief explanation of WHY you are choosing this route."
        )
        next: OptionsEnum = Field(
            description="The name of the next worker to route to, or __end__ if finished."
        )

    # Track consecutive routing to the same worker
    routing_history: list[str] = []

    def supervisor_node(state: MultiAgentState) -> Command:
        """The actual node function that runs in LangGraph."""
        logger.info(
            f"Supervisor evaluating state with {len(state['messages'])} messages..."
        )

        # Format the system prompt with the list of members
        formatted_prompt = system_prompt.format(members=", ".join(members))
        formatted_prompt += (
            "\n\nCRITICAL RULES:"
            "\n1. If the user's request has been completely fulfilled (e.g. a file was saved, "
            "a summary was provided), you MUST route to __end__."
            "\n2. Do NOT route to a worker just because it said 'let me know if you need anything'."
            "\n3. If a worker already completed its task successfully, do NOT send work back to it."
        )

        # FILTER: Prevent ContextWindowExceeded by omitting large ToolMessages
        filtered_messages = []
        for msg in state["messages"]:
            if msg.type == "tool":
                # Only include tool message if it's small, otherwise just note it happened
                if len(str(msg.content)) > 2000:
                    from langchain_core.messages import ToolMessage
                    filtered_messages.append(ToolMessage(
                        content=f"[Tool completed successfully. Output ({len(str(msg.content))} chars) hidden to save context window.]",
                        tool_call_id=msg.tool_call_id,
                        name=msg.name
                    ))
                else:
                    filtered_messages.append(msg)
            else:
                filtered_messages.append(msg)

        messages = [SystemMessage(content=formatted_prompt)] + filtered_messages

        # Force the LLM to output our exact Route schema
        response = llm.with_structured_output(Route).invoke(messages)

        next_node = response.next.value
        logger.info(f"Supervisor reason: {response.reason}")
        logger.info(f"Supervisor decision: Routing to -> {next_node}")

        # Safety: detect infinite loops by checking consecutive same-worker routing
        routing_history.append(next_node)
        if next_node != "__end__":
            # Count how many times the same worker appears at the tail of history
            consecutive = 0
            for past in reversed(routing_history):
                if past == next_node:
                    consecutive += 1
                else:
                    break
            if consecutive >= max_consecutive_same:
                logger.warning(
                    f"Safety limit reached: '{next_node}' was routed to "
                    f"{consecutive} times in a row. Forcing __end__."
                )
                next_node = "__end__"

        # LOUD SUPERVISOR: Inject explicit instruction if routing back to a worker
        if next_node != "__end__":
            manager_instruction = HumanMessage(
                content=f"Supervisor Directive: {response.reason}. Please address this.",
                name="Supervisor"
            )
            return Command(goto=next_node, update={"messages": [manager_instruction]})

        return Command(goto=next_node)

    return supervisor_node
