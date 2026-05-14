# Multi-Agent Web Research and Writing System

This project is a LangGraph multi-agent application for researching a topic, reviewing the research, and producing a written document. It uses specialized ReAct workers for research, analysis, and writing, plus a supervisor node that routes work through the pipeline.

The current supervisor is a hybrid router:

- It can use an LLM for the first route when no worker has acted yet.
- After a known worker finishes, it uses deterministic routing rules so the same completed worker is not called again by accident.

## Runtime Flow

Default pipeline:

```text
new user request
-> researcher_agent
-> analyst_agent
-> writer_agent
-> __end__
```

Revision path:

```text
analyst_agent + NEEDS_IMPROVEMENT
-> researcher_agent
-> analyst_agent
```

Completion path:

```text
analyst_agent + APPROVED
-> writer_agent
-> __end__
```

## Main Components

### `main.py`

`main.py` builds and runs the graph.

Responsibilities:

- Loads environment variables and API keys.
- Creates the shared LLM through `LLMFactory`.
- Defines worker names:
  - `researcher_agent`
  - `analyst_agent`
  - `writer_agent`
- Builds ReAct workers with their tool lists and prompts.
- Wraps each worker node so only new messages are returned to the graph.
- Tags worker `AIMessage` objects with the worker name.
- Stores important outputs in graph state.
- Compiles the graph with `MemorySaver`.
- Runs `app.stream(...)` in a command-line loop.

### `agents/supervisor.py`

The supervisor decides which node runs next.

For a brand-new request, the supervisor may call the LLM and request a structured route:

```json
{
  "reason": "Why this route was selected",
  "next": "researcher_agent | analyst_agent | writer_agent | __end__"
}
```

After a worker finishes, the supervisor no longer relies on the LLM. It reads the latest worker name from the last `AIMessage.name` and applies fixed routing:

| Last worker | Condition | Next node |
| --- | --- | --- |
| `researcher_agent` | always | `analyst_agent` |
| `analyst_agent` | output/state contains `NEEDS_IMPROVEMENT` | `researcher_agent` |
| `analyst_agent` | output/state contains `APPROVED` | `writer_agent` |
| `analyst_agent` | no clear status | `__end__` |
| `writer_agent` | always | `__end__` |

The supervisor also injects a `HumanMessage` named `Supervisor` when handing off to a worker:

```text
Supervisor Directive: <routing reason>. Please address this.
```

That directive is visible to the next worker and gives it immediate context for the task.

### `models/state.py`

The graph state is defined as `MultiAgentState`.

```python
class MultiAgentState(TypedDict):
    messages: Annotated[list, add_messages]
    research_report: str
    scraped_data: str
    document_draft: str
    approval_status: str
```

State fields:

| Field | Purpose |
| --- | --- |
| `messages` | Full LangChain message history. Uses LangGraph `add_messages`, so new messages append instead of replacing old ones. |
| `research_report` | Latest researcher summary/report content. |
| `scraped_data` | Intended location for large scraped payloads. |
| `document_draft` | Latest writer output. |
| `approval_status` | Analyst status such as `PENDING`, `APPROVED`, or `NEEDS_IMPROVEMENT`. |

## Worker Contract

Workers are LangGraph ReAct agents created with `create_react_agent`. They do not return plain strings directly to the graph. Each worker returns graph state updates containing new LangChain messages.

Current worker node return shape:

```python
{
    "messages": [AIMessage(...), ToolMessage(...), ...],
    "<optional_state_field>": "..."
}
```

The wrapper around each worker tags AI messages:

```python
AIMessage(
    content="I have completed the detailed research report...",
    name="researcher_agent"
)
```

This `name` field is important. The supervisor uses it to know which worker just completed a turn.

### Researcher Response

The researcher is responsible for search, scraping, source validation, and research synthesis.

Expected final response style:

```text
I have completed the research report.

Summary:
...

Sources:
- title, url, notes

Known gaps:
- ...
```

Graph update:

```python
{
    "messages": new_researcher_messages,
    "research_report": latest_researcher_ai_content
}
```

The researcher may call tools many times internally before returning control to the supervisor.

Researcher tools:

- `create_research_strategy`
- `validate_research_sources`
- `create_research_report`
- `analyze_research_completeness`
- `scrape_webpages`
- `search_tavily`
- `search_ddg`
- `enhanced_search_tavily`
- `search_wikipedia`
- `search_arxiv`
- `search_github`
- `search_stackoverflow`
- `get_tool_references`

### Analyst Response

The analyst reviews the research, identifies gaps, and decides whether writing can begin.

Required status language:

```text
APPROVED
```

or:

```text
NEEDS_IMPROVEMENT
```

Expected final response style:

```text
Approval Status: NEEDS_IMPROVEMENT

Critique:
- Missing coverage of ...
- Source quality concern ...

Recommended follow-up research:
- Search query ...
- Source type ...
```

Graph update:

```python
{
    "messages": new_analyst_messages,
    "approval_status": "APPROVED | NEEDS_IMPROVEMENT | PENDING"
}
```

Analyst tools:

- `validate_research_sources`
- `analyze_research_completeness`
- `get_tool_references`

### Writer Response

The writer turns approved research into a final document.

Expected final response style:

```text
Document created.

Location:
...

Summary:
...

Quality notes:
...
```

Graph update:

```python
{
    "messages": new_writer_messages,
    "document_draft": latest_writer_ai_content
}
```

Writer tools:

- `create_structured_document`
- `generate_research_summary`
- `validate_document_structure`
- `create_outline`
- `read_document`
- `write_document`
- `edit_document`

## Tool Call Contract

Tools are exposed to workers as LangChain tools. During a ReAct worker turn, the worker may produce an `AIMessage` with tool calls:

```python
AIMessage(
    content="I will use search_github because...",
    tool_calls=[
        {
            "name": "search_github",
            "args": {"query": "LangGraph checkpointing"},
            "id": "call_..."
        }
    ]
)
```

LangGraph executes the tool and appends a matching `ToolMessage`:

```python
ToolMessage(
    name="search_github",
    tool_call_id="call_...",
    content="..."
)
```

The worker continues this loop until it returns an `AIMessage` without tool calls. At that point, the worker node returns control to the supervisor.

## Research Source Schema

Research tools accept source JSON. The preferred flat shape is:

```json
[
  {
    "title": "LangGraph documentation",
    "url": "https://...",
    "content": "Relevant source content..."
  }
]
```

Grouped source shape is also accepted:

```json
[
  {
    "source": "GitHub",
    "details": [
      {
        "repo": "langchain-ai/langgraph",
        "url": "https://github.com/langchain-ai/langgraph",
        "stars": 32000,
        "updated": "2026-05-14"
      }
    ]
  }
]
```

The research tools normalize grouped records into flat source records before validation and report generation.

## Checkpointing and Memory

The graph is compiled with:

```python
memory = MemorySaver()
return builder.compile(checkpointer=memory)
```

`MemorySaver` stores graph state in process memory by `thread_id`.

Current runtime config:

```python
config = {
    "configurable": {"thread_id": "session_1"},
    "recursion_limit": 20
}
```

Important behavior:

- Reusing `"session_1"` means each prompt in the same process sees previous messages.
- This is useful for conversation continuity.
- It can confuse routing if old requests remain in the message history.
- Use a fresh `thread_id` per task if each user request should be isolated.
- `MemorySaver` is not persistent storage. State disappears when the Python process exits.

## Safety Limits

The supervisor tracks repeated route decisions with `max_consecutive_same`.

If the same worker is selected too many times in a row, the supervisor forces `__end__` to avoid infinite loops.

This is a guardrail, not the main routing mechanism. The main loop prevention comes from deterministic routing after worker completion.

## Logs

Logs are written to:

```text
logs/multiagent.log
```

The logging helper records:

- Worker tool choices.
- Tool arguments.
- Tool results.
- Worker AI message previews.
- Supervisor route reasons.
- Supervisor route decisions.

Example:

```text
[RESEARCHER] Tool chosen: search_github | Args: {...}
[RESEARCHER] Tool result (search_github): ...
Supervisor deterministic reason: researcher_agent completed its turn...
Supervisor decision: Routing to -> analyst_agent
```

## Project Structure

```text
.
├── agents/
│   └── supervisor.py
├── models/
│   ├── research_schemas.py
│   └── state.py
├── tools/
│   ├── document_generator.py
│   ├── document_tools.py
│   ├── enhanced_web_tools.py
│   ├── research_tools.py
│   └── web_tools.py
├── utils/
│   └── llm.py
├── main.py
├── mermaid.md
├── requirements.txt
└── README.md
```

## Getting Started

1. Create and activate a virtual environment.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Configure `.env` with required keys:

```text
OPENAI_API_KEY=...
AI_PROVIDER=...
TAVILY_API_KEY=...
```

4. Run the app:

```bash
python main.py
```

## Recommended Next Improvements

- Make analyst output a strict Pydantic schema instead of plain text status matching.
- Add `missing_topics`, and `recommended_queries` to `MultiAgentState`.
- Add `research_iterations` to state and cap follow-up research cycles.
- Use a fresh `thread_id` for each unrelated user task.
- Move large scraped text out of `messages` and into state fields or external storage.
