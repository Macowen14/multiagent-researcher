import os
import getpass
import logging
from dotenv import load_dotenv

from langgraph.graph import StateGraph, MessagesState, START
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import HumanMessage

from utils.llm import LLMFactory
from agents.supervisor import create_supervisor
from tools.web_tools import scrape_webpages, search_tavily, search_ddg
from tools.document_tools import create_outline, read_document, write_document, edit_document

# Setup Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
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

def print_stream(stream):
    """Helper to pretty print LangGraph streams so you can track state updates easily."""
    for step in stream:
        for node_name, state_update in step.items():
            print(f"\n==============================================")
            print(f"🔄 STATE UPDATE FROM: {node_name.upper()}")
            print(f"==============================================")
            
            if "messages" in state_update:
                messages = state_update["messages"]
                # Only print the most recently added message(s) for clarity
                if isinstance(messages, list) and len(messages) > 0:
                    latest_msg = messages[-1]
                    msg_type = latest_msg.__class__.__name__
                    
                    # Truncate content if it's too long
                    content = str(latest_msg.content)
                    if len(content) > 500:
                        content = content[:500] + "... [TRUNCATED]"
                        
                    print(f"[{msg_type}]:\n{content}")
                    
                    if hasattr(latest_msg, "tool_calls") and latest_msg.tool_calls:
                        print(f"🔧 Tool Calls: {latest_msg.tool_calls}")

def build_graph():
    """Builds and returns the LangGraph application."""
    # 1. Initialize LLM
    llm = LLMFactory.get_llm()
    
    # 2. Define our worker node names
    RESEARCHER = "researcher_agent"
    WRITER = "writer_agent"
    MEMBERS = [RESEARCHER, WRITER]
    
    # 3. Create the Supervisor Node
    supervisor_prompt = (
        "You are a supervisor managing a team of expert workers: {members}. "
        "The 'researcher_agent' is responsible for searching and scraping the web. "
        "The 'writer_agent' is responsible for creating outlines and editing documents. "
        "Given the conversation history and the user's request, decide which worker "
        "should act next to make progress. Each worker will perform a task and return results. "
        "If the user's request has been completely fulfilled, route to __end__."
    )
    supervisor_node = create_supervisor(
        llm=llm, 
        members=MEMBERS, 
        system_prompt=supervisor_prompt
    )
    
    # 4. Create Worker Nodes using prebuilt ReAct agents
    # The researcher gets web tools
    researcher_agent = create_react_agent(
        model=llm,
        tools=[scrape_webpages, search_tavily, search_ddg],
        state_modifier="You are a web researcher. Use your tools to find and read information online. Provide clear summaries of what you find."
    )
    
    # The writer gets document tools
    writer_agent = create_react_agent(
        model=llm,
        tools=[create_outline, read_document, write_document, edit_document],
        state_modifier="You are an expert writer and editor. Use your tools to read, write, and organize documents on the filesystem."
    )

    # Helper function to extract just the new messages from the prebuilt agent's state
    # because create_react_agent returns a full dict with {"messages": ...}
    def researcher_node(state: MessagesState):
        result = researcher_agent.invoke(state)
        # We only want to return the last message added by the agent to avoid duplicating history
        return {"messages": [result["messages"][-1]]}
        
    def writer_node(state: MessagesState):
        result = writer_agent.invoke(state)
        return {"messages": [result["messages"][-1]]}

    # 5. Build the Graph
    builder = StateGraph(MessagesState)
    
    # Add nodes
    builder.add_node("supervisor", supervisor_node)
    builder.add_node(RESEARCHER, researcher_node)
    builder.add_node(WRITER, writer_node)
    
    # Add edges
    builder.add_edge(START, "supervisor")
    # The supervisor dynamically routes, but the workers always route back to the supervisor
    builder.add_edge(RESEARCHER, "supervisor")
    builder.add_edge(WRITER, "supervisor")
    
    return builder.compile()

if __name__ == "__main__":
    setup_environment()
    
    app = build_graph()
    print("\n✅ LangGraph multi-agent application successfully built!\n")
    
    # Example execution
    user_input = "Can you scrape https://example.com and create an outline of it in example_outline.txt?"
    print(f"👤 USER REQUEST: {user_input}")
    
    stream = app.stream(
        {"messages": [HumanMessage(content=user_input)]},
        config={"recursion_limit": 20}
    )
    
    print_stream(stream)
