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

    def _last_message_worker(state: MultiAgentState) -> tuple[str | None, str]:
        """Return the worker that produced the latest message, if it is known."""
        if not state["messages"]:
            return None, ""

        latest_message = state["messages"][-1]
        latest_worker = getattr(latest_message, "name", None)
        if latest_worker in members:
            return latest_worker, str(getattr(latest_message, "content", "") or "")

        return None, ""

    def _route_after_worker(
        worker_name: str, worker_output: str, state: MultiAgentState
    ) -> tuple[str, str]:
        """Apply the fixed pipeline after a worker has completed a turn."""
        if worker_name == "researcher_agent":
            return (
                "analyst_agent",
                "researcher_agent completed its turn, so the next pipeline step is analyst_agent review.",
            )

        if worker_name == "analyst_agent":
            approval_status = state.get("approval_status", "")
            output_upper = worker_output.upper()
            if approval_status == "NEEDS_IMPROVEMENT" or "NEEDS_IMPROVEMENT" in output_upper:
                return (
                    "researcher_agent",
                    "analyst_agent found research gaps, so the next pipeline step is researcher_agent.",
                )
            if approval_status == "APPROVED" or "APPROVED" in output_upper:
                return (
                    "writer_agent",
                    "analyst_agent approved the research, so the next pipeline step is writer_agent.",
                )
            return (
                "__end__",
                "analyst_agent finished without an explicit APPROVED or NEEDS_IMPROVEMENT status, so stopping avoids an ambiguous routing loop.",
            )

        if worker_name == "writer_agent":
            return (
                "__end__",
                "writer_agent completed the writing step, so the workflow is finished.",
            )

        return "__end__", "Unknown worker completed; stopping the workflow."

    def _apply_safety_limit(next_node: str) -> str:
        routing_history.append(next_node)
        if next_node == "__end__":
            return next_node

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
            return "__end__"

        return next_node

    def supervisor_node(state: MultiAgentState) -> Command:
        """The actual node function that runs in LangGraph."""
        logger.info(
            f"Supervisor evaluating state with {len(state['messages'])} messages..."
        )

        # Reset routing history if the latest message is a new Human prompt
        if (
            state["messages"]
            and isinstance(state["messages"][-1], HumanMessage)
            and state["messages"][-1].name != "Supervisor"
        ):
            logger.info("New user request detected. Resetting routing safety history.")
            routing_history.clear()

        last_worker, worker_output = _last_message_worker(state)
        if last_worker:
            next_node, reason = _route_after_worker(last_worker, worker_output, state)
            logger.info(f"Supervisor deterministic reason: {reason}")
            logger.info(f"Supervisor decision: Routing to -> {next_node}")
            next_node = _apply_safety_limit(next_node)

            if next_node != "__end__":
                manager_instruction = HumanMessage(
                    content=f"Supervisor Directive: {reason}. Please address this.",
                    name="Supervisor",
                )
                return Command(
                    goto=next_node, update={"messages": [manager_instruction]}
                )

            return Command(goto=next_node)

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

                    filtered_messages.append(
                        ToolMessage(
                            content=(
                                "[Tool completed successfully. Output "
                                f"({len(str(msg.content))} chars) hidden to save context window.]"
                            ),
                            tool_call_id=msg.tool_call_id,
                            name=msg.name,
                        )
                    )
                else:
                    filtered_messages.append(msg)
            else:
                filtered_messages.append(msg)

        messages = [SystemMessage(content=formatted_prompt)] + filtered_messages

        # Inject recency reminder so the LLM doesn't get confused by early messages
        if state["messages"] and state["messages"][-1].name in members:
            last_agent = state["messages"][-1].name
            messages.append(
                SystemMessage(
                    content=(
                        f"SYSTEM REMINDER: '{last_agent}' just finished its turn. "
                        f"DO NOT route back to '{last_agent}'. Route to the next "
                        "logical agent or to __end__."
                    )
                )
            )

        # Force the LLM to output our exact Route schema
        response = llm.with_structured_output(Route).invoke(messages)

        next_node = response.next.value
        logger.info(f"Supervisor reason: {response.reason}")
        logger.info(f"Supervisor decision: Routing to -> {next_node}")

        next_node = _apply_safety_limit(next_node)

        # LOUD SUPERVISOR: Inject explicit instruction if routing back to a worker
        if next_node != "__end__":
            manager_instruction = HumanMessage(
                content=f"Supervisor Directive: {response.reason}. Please address this.",
                name="Supervisor",
            )
            return Command(goto=next_node, update={"messages": [manager_instruction]})

        return Command(goto=next_node)

    return supervisor_node
