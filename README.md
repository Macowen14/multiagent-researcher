# Multi-Agent Web Research & Writing System

This project is a highly robust, production-ready multi-agent application built on the **LangGraph** framework. It implements a **Hierarchical Supervisor Routing Architecture** where a central LLM orchestrates specialized sub-agents (ReAct worker nodes) using strictly typed Pydantic models, custom state management, and memory checkpointing.

## Architectural Deep Dive

### 1. State Management (`MultiAgentState`)
The application uses a custom `TypedDict` for its graph state (defined in `models/state.py`).
- **`messages`**: A list of LangChain `BaseMessage` objects (`HumanMessage`, `AIMessage`, `ToolMessage`). This is managed by the LangGraph `add_messages` reducer which automatically appends new messages rather than overwriting them.
- **Data Fields**: Custom fields like `scraped_data`, `research_report`, and `approval_status` are defined to decouple large data payloads from the conversational chat history, drastically reducing token usage and preventing `ContextWindowExceeded` errors.

### 2. The Supervisor Node
Located in `agents/supervisor.py`, the Supervisor acts as the routing brain. 
- **Silent/Loud Routing**: The Supervisor reads the `MultiAgentState`, applies rules from its System Prompt, and uses `llm.with_structured_output` to force the generation of a JSON `Route` object. It does not execute tools.
- **Explicit Handoffs**: To solve "Identity Confusion" between agents, the Supervisor injects a `HumanMessage` named "Supervisor" into the state before routing. This explicit directive ensures the worker knows exactly what to fix.
- **State Filtering**: Before invoking the LLM, the Supervisor filters out large `ToolMessage` payloads (e.g., massive scraped HTML) replacing them with summaries. This preserves the context window.

### 3. Worker Nodes (ReAct Agents)
The Researcher, Analyst, and Writer are implemented via LangGraph's `create_react_agent`.
- **Execution Loop**: Workers execute internal `while tool_calls` loops. They parse tool output, decide on the next tool, and aggregate data. They only return control to the Supervisor when they generate an `AIMessage` without a tool call.
- **Tool Binding**: Tools are defined using the `@tool` decorator or Pydantic `BaseModel`s. LangChain compiles these into JSON schemas sent to the LLM (OpenAI/Ollama).

### 4. Checkpointing and Memory
The `StateGraph` is compiled with `MemorySaver()`. In `main.py`, the `app.stream()` execution is bound to a `thread_id`. This grants the multi-agent system thread-level persistent memory, allowing conversational continuity across multiple user prompts.

## Project Structure

* **`main.py`**: The entry point. Initializes the graph, sets up Memory Checkpointing, and runs the stream loop.
* **`models/state.py`**: Contains the `MultiAgentState` schema.
* **`agents/supervisor.py`**: Contains the `create_supervisor()` factory function and intelligent routing logic.
* **`tools/`**: 
  * `web_tools.py` & `enhanced_web_tools.py`: Search engines (Tavily, DDG, Github, arXiv) and safe DOM scraping (try/except wrapped).
  * `document_tools.py`: Safe atomic file writing tools.
* **`utils/llm.py`**: Factory for seamless switching between OpenAI and Ollama.

## Getting Started

1. Set up `.env` with API keys (`OPENAI_API_KEY`, `TAVILY_API_KEY`, etc.).
2. Run the application:
   ```bash
   python main.py
   ```
3. Logs are generated dynamically in the `logs/multiagent.log` file.
