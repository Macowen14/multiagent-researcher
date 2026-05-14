import os
import getpass
import logging
from dotenv import load_dotenv
from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import create_react_agent
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.messages import HumanMessage

from models.state import MultiAgentState

from utils.llm import LLMFactory
from agents.supervisor import create_supervisor
from tools.web_tools import scrape_webpages, search_tavily, search_ddg
from tools.enhanced_web_tools import (
    enhanced_search_tavily,
    search_wikipedia,
    search_arxiv,
    search_github,
    search_stackoverflow,
    get_tool_references,
)
from tools.research_tools import (
    create_research_strategy,
    validate_research_sources,
    create_research_report,
    analyze_research_completeness,
)
from models.research_schemas import (
    ResearchStrategy,
    ResearchReport,
    AnalysisReport,
    ValidationResponse,
    SearchToolType,
    ContentType,
    ResearchQuality,
)
from tools.document_tools import (
    create_outline,
    read_document,
    write_document,
    edit_document,
)
from tools.document_generator import (
    create_structured_document,
    generate_research_summary,
    validate_document_structure,
)

# Setup Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/multiagent.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Reduce noise from underlying HTTP libraries during execution
logging.getLogger("httpx").setLevel(logging.WARNING)


def setup_environment():
    """Ensure all required environment variables are set."""
    load_dotenv()

    # Define a default USER_AGENT to avoid LangChain community warnings.
    os.environ.setdefault("USER_AGENT", "langchain-community/1.0")

    envlist = ["OPENAI_API_KEY", "AI_PROVIDER", "TAVILY_API_KEY"]
    for env_var in envlist:
        if not os.getenv(env_var):
            value = getpass.getpass(f"Enter value for {env_var}: ")
            os.environ[env_var] = value


def build_graph():
    """Builds and returns the LangGraph application."""
    # 1. Initialize LLM
    llm = LLMFactory.get_llm()

    # 2. Define our worker node names
    RESEARCHER = "researcher_agent"
    WRITER = "writer_agent"
    ANALYST = "analyst_agent"
    MEMBERS = [RESEARCHER, ANALYST, WRITER]

    # 3. Create the Supervisor Node
    supervisor_prompt = (
        "You are an INTELLIGENT supervisor managing a team of expert workers: {members}. "
        "The 'researcher_agent' is responsible for searching and scraping the web using multiple specialized tools. "
        "The 'analyst_agent' is responsible for analyzing research, identifying gaps, and planning next steps. "
        "The 'writer_agent' is responsible for creating outlines and editing documents. "
        "\nINTELLIGENT ROUTING STRATEGY:\n"
        "1. For ALL new research requests, you MUST route to 'researcher_agent' first.\n"
        "2. Do NOT route to 'writer_agent' or 'analyst_agent' initially.\n"
        "3. Once 'researcher_agent' has provided data, route to 'analyst_agent' for review.\n"
        "4. ONLY route to 'writer_agent' once 'analyst_agent' has explicitly approved the research.\n"
        "5. If 'analyst_agent' finds gaps, route back to 'researcher_agent'.\n\n"
        "Given the conversation history and the user's request, decide which worker "
        "should act next to make progress. Each worker will perform a task and return results. "
        "If the user's request has been completely fulfilled, route to __end__."
    )
    supervisor_node = create_supervisor(
        llm=llm, members=MEMBERS, system_prompt=supervisor_prompt
    )

    # Shared instruction appended to all worker prompts so the LLM always explains its tool choice
    TOOL_LOGGING_INSTRUCTION = (
        "\n\nBefore calling any tool, briefly state: "
        "1) Which tool you are choosing, "
        "2) Why you chose it, and "
        "3) Your confidence score (1-10) that this is the right tool for the job."
    )

    # 4. Create Worker Nodes using prebuilt ReAct agents
    # The researcher gets enhanced web tools with intelligent decision-making
    researcher_agent = create_react_agent(
        model=llm,
        tools=[
            create_research_strategy,
            validate_research_sources,
            create_research_report,
            analyze_research_completeness,
            scrape_webpages,
            search_tavily,
            search_ddg,
            enhanced_search_tavily,
            search_wikipedia,
            search_arxiv,
            search_github,
            search_stackoverflow,
            get_tool_references,
        ],
        prompt=(
            "You are an EXPERT RESEARCHER with Pydantic-structured output capabilities. "
            "Your role is to conduct comprehensive, deep research using structured methodologies.\n\n"
            "RESEARCH WORKFLOW:\n"
            "1. **STRATEGY PHASE**: Use `create_research_strategy` to plan your approach\n"
            "2. **EXECUTION PHASE**: Use appropriate search tools based on the strategy\n"
            "3. **VALIDATION PHASE**: Use `validate_research_sources` to ensure quality\n"
            "4. **SYNTHESIS PHASE**: Use `create_research_report` to structure findings\n"
            "5. **COMPLETENESS CHECK**: Use `analyze_research_completeness` to identify gaps\n\n"
            "AVAILABLE TOOLS:\n"
            "- **Strategy Tools**: create_research_strategy, validate_research_sources\n"
            "- **Search Tools**: enhanced_search_tavily, search_wikipedia, search_arxiv, search_github, search_stackoverflow\n"
            "- **Analysis Tools**: create_research_report, analyze_research_completeness\n"
            "- **Reference Tools**: get_tool_references, scrape_webpages\n\n"
            "RESEARCH PRINCIPLES:\n"
            "1. ALWAYS start with `create_research_strategy` to plan your approach\n"
            "2. NEVER use pre-trained knowledge - ONLY use tool results\n"
            "3. Cross-reference findings from multiple sources\n"
            "4. Validate sources using `validate_research_sources`\n"
            "5. Structure all outputs using Pydantic models\n"
            "6. Include confidence scores and quality assessments\n"
            "7. Identify and explicitly state research gaps\n"
            "8. Provide specific recommendations for additional research\n\n"
            "QUALITY STANDARDS:\n"
            "- Minimum 5 high-quality sources per topic\n"
            "- Source diversity (academic, practical, documentation)\n"
            "- Recent sources (within 2 years when relevant)\n"
            "- Proper attribution and reference links\n"
            "- Confidence scoring for all claims\n\n"
            "Your outputs should be structured, comprehensive, and ready for analysis by the analyst agent."
            + TOOL_LOGGING_INSTRUCTION
        ),
    )

    # The writer gets enhanced document generation tools with Pydantic-structured outputs
    writer_agent = create_react_agent(
        model=llm,
        tools=[
            create_structured_document,
            generate_research_summary,
            validate_document_structure,
            create_outline,
            read_document,
            write_document,
            edit_document,
        ],
        prompt=(
            "You are an EXPERT DOCUMENT CREATOR with Pydantic-structured output capabilities. "
            "Your role is to transform research into professional, well-structured documents.\n\n"
            "DOCUMENT CREATION WORKFLOW:\n"
            "1. **STRUCTURE PLANNING**: Use `create_structured_document` for organized output\n"
            "2. **CONTENT GENERATION**: Create comprehensive documents with proper sections\n"
            "3. **QUALITY VALIDATION**: Use `validate_document_structure` to ensure quality\n"
            "4. **SUMMARY CREATION**: Use `generate_research_summary` for executive summaries\n"
            "5. **FILE MANAGEMENT**: Use document tools for file operations\n\n"
            "DOCUMENT TYPES:\n"
            "- **Research Reports**: Comprehensive analysis with methodology and findings\n"
            "- **Comparative Analysis**: Side-by-side comparisons with pros/cons\n"
            "- **Executive Summaries**: Brief overviews for decision-makers\n"
            "- **Technical Documentation**: Detailed technical guides and references\n\n"
            "STRUCTURE REQUIREMENTS:\n"
            "1. **Never hallucinate** - ONLY use research provided in chat history\n"
            "2. **Proper attribution** - All claims must be linked to sources\n"
            "3. **Logical organization** - Clear sections and subsections\n"
            "4. **Professional formatting** - Consistent markdown structure\n"
            "5. **Comprehensive references** - Complete citation lists\n"
            "6. **Quality validation** - Ensure document meets standards\n\n"
            "OUTPUT STANDARDS:\n"
            "- Executive summary with key findings\n"
            "- Detailed methodology and approach\n"
            "- Well-structured findings with evidence\n"
            "- Proper conclusions and recommendations\n"
            "- Complete reference sections\n"
            "- Quality assessment scores\n\n"
            "Always validate documents before finalizing and provide clear confirmation "
            "of what was created, where it's saved, and its quality score."
            + TOOL_LOGGING_INSTRUCTION
        ),
    )

    # The analyst gets enhanced analysis tools with Pydantic-structured outputs
    analyst_agent = create_react_agent(
        model=llm,
        tools=[
            validate_research_sources,
            analyze_research_completeness,
            get_tool_references,
        ],
        prompt=(
            "You are an EXPERT RESEARCH ANALYST with Pydantic-structured analysis capabilities. "
            "Your role is to conduct rigorous quality assessment and gap analysis.\n\n"
            "ANALYSIS WORKFLOW:\n"
            "1. **QUALITY ASSESSMENT**: Use `validate_research_sources` to evaluate source quality\n"
            "2. **COMPLETENESS ANALYSIS**: Use `analyze_research_completeness` to identify gaps\n"
            "3. **STRUCTURED EVALUATION**: Create structured analysis reports\n"
            "4. **RECOMMENDATION ENGINE**: Provide specific, actionable recommendations\n\n"
            "ANALYSIS FRAMEWORK:\n"
            "- **Source Quality**: Assess reliability, relevance, and diversity\n"
            "- **Coverage Analysis**: Evaluate topic completeness and depth\n"
            "- **Gap Identification**: Pinpoint missing information areas\n"
            "- **Confidence Scoring**: Rate reliability of findings\n"
            "- **Action Planning**: Recommend specific next steps\n\n"
            "EVALUATION CRITERIA:\n"
            "1. **Source Diversity**: Multiple source types (academic, practical, documentation)\n"
            "2. **Information Depth**: Comprehensive coverage beyond surface-level\n"
            "3. **Cross-Validation**: Information corroborated by multiple sources\n"
            "4. **Recency**: Current information when relevant\n"
            "5. **Attribution**: Proper source citation and references\n\n"
            "OUTPUT REQUIREMENTS:\n"
            "- Use structured analysis with clear scoring\n"
            "- Provide specific recommendations for improvement\n"
            "- Include confidence levels for all assessments\n"
            "- Clearly state approval status (APPROVED/NEEDS_IMPROVEMENT)\n"
            "- Reference tools used for analysis\n\n"
            "When research meets quality standards, approve for writer agent. "
            "When gaps exist, provide specific tool recommendations and search queries."
            + TOOL_LOGGING_INSTRUCTION
        ),
    )

    def _log_new_messages(node_name: str, new_messages):
        """Inspect and log every tool call and AI response from a worker's new messages."""
        for msg in new_messages:
            msg_type = msg.__class__.__name__

            # Log tool calls made by the AI
            if hasattr(msg, "tool_calls") and msg.tool_calls:
                for tc in msg.tool_calls:
                    logger.info(
                        f"[{node_name}] 🔧 Tool chosen: '{tc['name']}' | "
                        f"Args: {tc['args']}"
                    )

            # Log the AI's reasoning text (contains the confidence score)
            if msg_type == "AIMessage" and msg.content:
                content_preview = str(msg.content).replace("\n", " ")[:100]
                logger.info(f"[{node_name}] 💬 {msg_type}: {content_preview}...")

            # Log tool results
            if msg_type == "ToolMessage":
                result_preview = str(msg.content).replace("\n", " ")[:50]
                logger.info(
                    f"[{node_name}] 📦 Tool result ({msg.name}): {result_preview}..."
                )

    # Helper function to extract just the new messages from the prebuilt agent's state
    def researcher_node(state: MultiAgentState):
        existing_count = len(state["messages"])
        result = researcher_agent.invoke(state)
        new_msgs = result["messages"][existing_count:]
        _log_new_messages("RESEARCHER", new_msgs)
        return {"messages": new_msgs}

    def writer_node(state: MultiAgentState):
        existing_count = len(state["messages"])
        result = writer_agent.invoke(state)
        new_msgs = result["messages"][existing_count:]
        _log_new_messages("WRITER", new_msgs)
        return {"messages": new_msgs}

    def analyst_node(state: MultiAgentState):
        existing_count = len(state["messages"])
        result = analyst_agent.invoke(state)
        new_msgs = result["messages"][existing_count:]
        _log_new_messages("ANALYST", new_msgs)
        return {"messages": new_msgs}

    # 5. Build the Graph
    builder = StateGraph(MultiAgentState)

    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node(RESEARCHER, researcher_node)
    builder.add_node(WRITER, writer_node)
    builder.add_node(ANALYST, analyst_node)

    # Add edges
    builder.add_edge(START, "supervisor")
    # The supervisor dynamically routes, but the workers always route back to the supervisor
    builder.add_edge(RESEARCHER, "supervisor")
    builder.add_edge(WRITER, "supervisor")
    builder.add_edge(ANALYST, "supervisor")

    memory = MemorySaver()
    return builder.compile(checkpointer=memory)


if __name__ == "__main__":
    setup_environment()

    app = build_graph()
    print("\n✅ LangGraph multi-agent application successfully built!")
    print("Type 'exit' or 'quit' to stop the application.\n")

    try:
        while True:
            user_input = input("\n👤 USER REQUEST: ").strip()
            if not user_input:
                continue

            if user_input.lower() in ["exit", "quit"]:
                print("Exiting application. Goodbye!")
                break

            logger.info("Starting new workflow...")
            config = {"configurable": {"thread_id": "session_1"}, "recursion_limit": 20}
            for step in app.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config=config,
            ):
                for node_name, state_update in step.items():
                    logger.info(f"Node '{node_name.upper()}' completed its task.")

            logger.info("🎉 Workflow completed successfully!")

    except KeyboardInterrupt:
        print("\n\n🛑 Execution interrupted by user. Shutting down gracefully...")
