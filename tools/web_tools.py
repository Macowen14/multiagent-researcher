import logging
from typing import List, Annotated
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader

# Import Tavily search tool (requires TAVILY_API_KEY environment variable)
from langchain_community.tools.tavily_search import TavilySearchResults

# Import DuckDuckGo search (the user confirmed module is ddgs)
from ddgs import DDGS

logger = logging.getLogger(__name__)


@tool(
    "scrape_webpages", description="Scrapes the full text content from a list of URLs."
)
def scrape_webpages(urls: List[str]) -> str:
    """Use WebBaseLoader to scrape the content of the provided URLs and return the combined text."""
    logger.info(f"Scraping webpages: {urls}")

    loader = WebBaseLoader(urls)
    documents = loader.load()
    logger.info(f"Loaded {len(documents)} documents from the provided URLs.")

    # Join with double newline to keep pages distinct
    content = "\n\n".join([doc.page_content for doc in documents])
    logger.info("Webpage scraping complete.")
    return content


@tool(
    "search_tavily",
    description="Search the web using Tavily. Good for factual, recent, or detailed searches.",
)
def search_tavily(query: Annotated[str, "The search query to look up."]) -> str:
    """Uses Tavily search engine to search the web and return the top results."""
    logger.info(f"Tavily searching for: '{query}'")

    # Initialize the search tool (defaults to 3 results)
    tavily_tool = TavilySearchResults(max_results=3)

    # Invoke the search
    results = tavily_tool.invoke({"query": query})

    logger.info("Tavily search complete.")

    # Format the results into a readable string for the LLM
    if isinstance(results, list):
        formatted_results = []
        for r in results:
            url = r.get("url", "Unknown URL")
            content = r.get("content", "No content provided")
            formatted_results.append(f"Source: {url}\nContent: {content}")
        return "\n\n---\n\n".join(formatted_results)

    # Fallback if Tavily returns a raw string or dict error
    return str(results)


@tool(
    "search_ddg",
    description="Search the web using DuckDuckGo. Good for quick, general, or broad queries.",
)
def search_ddg(query: Annotated[str, "The search query to look up."]) -> str:
    """Uses DuckDuckGo to search the web and return the top results."""
    logger.info(f"DuckDuckGo searching for: '{query}'")

    # DDGS().text() returns an iterator/list of result dictionaries
    # Limit to 3 results to save context window space
    try:
        results = list(DDGS().text(query, max_results=3))

        if not results:
            return "No results found on DuckDuckGo."

        formatted_results = []
        for r in results:
            title = r.get("title", "No Title")
            body = r.get("body", "No Content")
            href = r.get("href", "No URL")
            formatted_results.append(f"Title: {title}\nSource: {href}\nContent: {body}")

        logger.info("DuckDuckGo search complete.")
        return "\n\n---\n\n".join(formatted_results)

    except Exception as e:
        logger.error(f"DuckDuckGo search failed: {e}")
        return f"DuckDuckGo search failed. Error: {str(e)}"
