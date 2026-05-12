# Multi-Agent Web Research & Writing System

This project is a highly robust, LangGraph-based multi-agent application. It uses an **Intelligent Supervisor** architecture to orchestrate three specialized workers: a **Researcher Agent**, an **Analyst Agent**, and a **Writer Agent**.

## Project Structure

* **`main.py`**: The entry point that sets up the LangGraph state graph, initializes the LLM, connects the agents, and runs an interactive streaming loop for processing user requests and logging operations.
* **`agents/supervisor.py`**: Contains a highly reusable factory function `create_supervisor()`. This node uses Pydantic schemas and `with_structured_output` to physically restrict the LLM to only make valid routing decisions, demanding reasoning and enforcing safety limits.
* **`tools/web_tools.py`**: A suite of tools for extracting information from the web:
  * `search_tavily`, `search_ddg`, `scrape_webpages`
* **`tools/enhanced_web_tools.py`**: Advanced domain-specific search integrations:
  * `enhanced_search_tavily`, `search_wikipedia`, `search_arxiv`, `search_github`, `search_stackoverflow`, `get_tool_references`
* **`tools/document_tools.py`**: A suite of tools for writing to the local filesystem:
  * `create_outline`, `write_document`, `read_document`, `edit_document`
* **`utils/util.py`**: A custom utility script that handles **Atomic File Locking**.
* **`utils/llm.py`**: An LLM factory pattern supporting OpenAI and Ollama.

## How the Agents Work Together

1. **Supervisor Node**: The "Brain" of the operation. It dynamically routes tasks between agents using an intelligent strategy:
   - *New requests* go to the Researcher.
   - *Completed research* goes to the Analyst.
   - *Approved analysis* goes to the Writer.
   - *Insufficient research* routes back to the Researcher.
2. **Researcher Agent**: Equipped with diverse web and academic tools. It performs deep, exhaustive searches across Wikipedia, arXiv, GitHub, and general search engines.
3. **Analyst Agent**: The quality controller. It reviews the research for gaps, ensures proper source attribution, and plans the document structure.
4. **Writer Agent**: The content creator. Organizes the finalized research and safely writes it to the local filesystem.

## Getting Started

1. Make sure your `.env` has the necessary keys (e.g. `OPENAI_API_KEY`, `TAVILY_API_KEY`).
2. Run the application:
   ```bash
   python main.py
   ```
