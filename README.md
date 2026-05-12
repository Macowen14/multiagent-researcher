# Multi-Agent Web Research & Writing System

This project is a highly robust, LangGraph-based multi-agent application. It uses a **Supervisor** architecture to orchestrate two specialized workers: a **Researcher Agent** and a **Writer Agent**.

## Project Structure

* **`main.py`**: The entry point that sets up the LangGraph state graph, initializes the LLM, connects the agents, and runs a clean streaming loop to format execution logs to the terminal.
* **`agents/supervisor.py`**: Contains a highly reusable factory function `create_supervisor()`. This node uses Pydantic schemas and `with_structured_output` to physically restrict the LLM to only make valid routing decisions.
* **`tools/web_tools.py`**: A suite of tools for extracting information from the web:
  * `search_tavily`: High-quality, AI-optimized semantic search.
  * `search_ddg`: DuckDuckGo search for quick, broad queries.
  * `scrape_webpages`: LangChain's WebBaseLoader to extract full raw text from websites.
* **`tools/document_tools.py`**: A suite of tools for writing to the local filesystem:
  * `create_outline`, `write_document`, `read_document`, `edit_document`.
* **`utils/util.py`**: A custom utility script that handles **Atomic File Locking**. It uses `os.open` with `O_CREAT | O_EXCL` flags inside a context manager to guarantee that multiple agents cannot corrupt or overwrite the same file at the exact same millisecond.
* **`utils/llm.py`**: An LLM factory pattern. It reads the `AI_PROVIDER` environment variable and automatically spins up and returns either an OpenAI or an Ollama chat model.

## How the Agents Work Together

1. **Supervisor Node**: The "Brain" of the operation. It reads the conversation history and dynamically decides who should work next. It can route to either the Researcher, the Writer, or finish the job (`__end__`).
2. **Researcher Agent**: Has access to the `web_tools`. When prompted by the supervisor, it searches the internet, reads webpages, and summarizes findings. It then reports back to the supervisor.
3. **Writer Agent**: Has access to the `document_tools`. When prompted by the supervisor, it organizes the research provided into outlines and drafts them into files safely. It then reports back to the supervisor.

## Getting Started

1. Make sure your `.env` has the necessary keys (e.g. `OPENAI_API_KEY`, `TAVILY_API_KEY`). If you forget them, the script will securely prompt you for them at runtime!
2. Run the application:
   ```bash
   python main.py
   ```
