import logging
import requests
import json
from typing import List, Annotated, Dict, Any
from langchain_core.tools import tool
from langchain_community.document_loaders import WebBaseLoader

from langchain_community.tools.tavily_search import TavilySearchResults
from ddgs import DDGS

logger = logging.getLogger(__name__)

# Reference links for tools and information
TOOL_REFERENCES = {
    "tavily": {
        "url": "https://tavily.com/",
        "description": "Tavily AI - Advanced search API for accurate, real-time web search results",
        "docs": "https://docs.tavily.com/docs/python-sdk/tavily-search"
    },
    "duckduckgo": {
        "url": "https://duckduckgo.com/",
        "description": "DuckDuckGo - Privacy-focused search engine",
        "docs": "https://duckduckgo.com/help/results/features"
    },
    "wikipedia": {
        "url": "https://www.wikipedia.org/",
        "description": "Wikipedia - Free encyclopedia with comprehensive articles",
        "docs": "https://en.wikipedia.org/wiki/Help:Contents"
    },
    "arxiv": {
        "url": "https://arxiv.org/",
        "description": "arXiv - Open access repository for scientific papers",
        "docs": "https://arxiv.org/help/api"
    },
    "github": {
        "url": "https://github.com/",
        "description": "GitHub - Code hosting and version control platform",
        "docs": "https://docs.github.com/en/rest"
    },
    "stackoverflow": {
        "url": "https://stackoverflow.com/",
        "description": "Stack Overflow - Q&A platform for programmers",
        "docs": "https://api.stackexchange.com/docs"
    }
}

@tool("get_tool_references")
def get_tool_references(
    tool_name: Annotated[str, "Name of the tool to get references for (e.g., 'tavily', 'wikipedia')"]
) -> str:
    """Get reference links and documentation for tools and information sources."""
    tool_name = tool_name.lower()
    
    if tool_name in TOOL_REFERENCES:
        ref = TOOL_REFERENCES[tool_name]
        return f"""**{tool_name.title()} References:**
📖 Description: {ref['description']}
🌐 Website: {ref['url']}
📚 Documentation: {ref['docs']}
"""
    else:
        available_tools = ", ".join(TOOL_REFERENCES.keys())
        return f"Tool '{tool_name}' not found. Available tools: {available_tools}"

@tool("search_wikipedia")
def search_wikipedia(
    query: Annotated[str, "The search query to look up on Wikipedia"],
    max_results: Annotated[int, "Maximum number of results to return"] = 3
) -> str:
    """Search Wikipedia for comprehensive encyclopedia articles."""
    logger.info(f"Wikipedia searching for: '{query}'")
    
    try:
        import wikipedia
        wikipedia.set_lang("en")
        
        # Search for pages
        search_results = wikipedia.search(query, results=max_results)
        
        if not search_results:
            return "No Wikipedia articles found for this query."
        
        formatted_results = []
        for title in search_results:
            try:
                # Get page summary
                page = wikipedia.page(title, auto_suggest=False)
                summary = page.summary[:500] + "..." if len(page.summary) > 500 else page.summary
                formatted_results.append(
                    f"Title: {title}\n"
                    f"URL: {page.url}\n"
                    f"Summary: {summary}\n"
                    f"References: 📚 [Wikipedia]({page.url}) | 🌐 {TOOL_REFERENCES['wikipedia']['url']}"
                )
            except Exception as e:
                formatted_results.append(f"Title: {title}\nError retrieving page: {str(e)}")
        
        logger.info("Wikipedia search complete.")
        return "\n\n---\n\n".join(formatted_results)
        
    except ImportError:
        return "Wikipedia library not installed. Install with: pip install wikipedia"
    except Exception as e:
        logger.error(f"Wikipedia search failed: {e}")
        return f"Wikipedia search failed. Error: {str(e)}"

@tool("search_arxiv")
def search_arxiv(
    query: Annotated[str, "The search query for academic papers"],
    max_results: Annotated[int, "Maximum number of papers to return"] = 3
) -> str:
    """Search arXiv for academic research papers and articles."""
    logger.info(f"arXiv searching for: '{query}'")
    
    try:
        import arxiv
        
        # Search for papers
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance
        )
        
        formatted_results = []
        for paper in search.results():
            summary = paper.summary[:300] + "..." if len(paper.summary) > 300 else paper.summary
            formatted_results.append(
                f"Title: {paper.title}\n"
                f"Authors: {', '.join(author.name for author in paper.authors[:3])}\n"
                f"Published: {paper.published.strftime('%Y-%m-%d')}\n"
                f"Summary: {summary}\n"
                f"URL: {paper.entry_id}\n"
                f"References: 📚 [arXiv]({paper.entry_id}) | 🌐 {TOOL_REFERENCES['arxiv']['url']}"
            )
        
        if not formatted_results:
            return "No arXiv papers found for this query."
        
        logger.info("arXiv search complete.")
        return "\n\n---\n\n".join(formatted_results)
        
    except ImportError:
        return "arXiv library not installed. Install with: pip install arxiv"
    except Exception as e:
        logger.error(f"arXiv search failed: {e}")
        return f"arXiv search failed. Error: {str(e)}"

@tool("search_github")
def search_github(
    query: Annotated[str, "The search query for GitHub repositories"],
    sort: Annotated[str, "Sort order: 'stars', 'forks', 'updated'"] = "stars",
    max_results: Annotated[int, "Maximum number of repositories to return"] = 3
) -> str:
    """Search GitHub for relevant code repositories."""
    logger.info(f"GitHub searching for: '{query}' (sorted by {sort})")
    
    try:
        # Using GitHub's search API via requests
        url = "https://api.github.com/search/repositories"
        params = {
            "q": query,
            "sort": sort,
            "order": "desc",
            "per_page": max_results
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("items"):
            return "No GitHub repositories found for this query."
        
        formatted_results = []
        for repo in data["items"]:
            description = repo.get("description")
            if description is None:
                description = "No description available"
            description = description[:200] + "..." if len(description) > 200 else description
            
            formatted_results.append(
                f"Repository: {repo['full_name']}\n"
                f"Description: {description}\n"
                f"Stars: {repo['stargazers_count']} | Forks: {repo['forks_count']}\n"
                f"Language: {repo.get('language', 'Unknown')}\n"
                f"Updated: {repo['updated_at'][:10]}\n"
                f"URL: {repo['html_url']}\n"
                f"References: 🔗 [GitHub]({repo['html_url']}) | 🌐 {TOOL_REFERENCES['github']['url']}"
            )
        
        logger.info("GitHub search complete.")
        return "\n\n---\n\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"GitHub search failed: {e}")
        return f"GitHub search failed. Error: {str(e)}"

@tool("enhanced_search_tavily")
def enhanced_search_tavily(
    query: Annotated[str, "The search query to look up."],
    include_raw: Annotated[bool, "Include raw content for deeper analysis"] = False,
    max_results: Annotated[int, "Maximum number of results"] = 5
) -> str:
    """Enhanced Tavily search with options for deeper analysis and references."""
    logger.info(f"Enhanced Tavily searching for: '{query}' (max_results={max_results})")
    
    try:
        # Initialize the search tool
        tavily_tool = TavilySearchResults(max_results=max_results)
        
        # Invoke the search
        results = tavily_tool.invoke({"query": query})
        
        logger.info("Enhanced Tavily search complete.")
        
        # Format the results with references
        if isinstance(results, list):
            formatted_results = []
            for i, r in enumerate(results, 1):
                url = r.get("url", "Unknown URL")
                content = r.get("content", "No content provided")
                title = r.get("title", f"Result {i}")
                
                formatted_result = f"""Title: {title}
Source: {url}
Content: {content}
References: 🔗 [Source]({url}) | 🛠️ [Tavily Docs]({TOOL_REFERENCES['tavily']['docs']}) | 🌐 [Tavily]({TOOL_REFERENCES['tavily']['url']})"""
                
                formatted_results.append(formatted_result)
            
            return "\n\n---\n\n".join(formatted_results)
        
        # Fallback if Tavily returns a raw string or dict error
        return str(results)
        
    except Exception as e:
        logger.error(f"Enhanced Tavily search failed: {e}")
        return f"Enhanced Tavily search failed. Error: {str(e)}"

@tool("search_stackoverflow")
def search_stackoverflow(
    query: Annotated[str, "The programming question to search for"],
    max_results: Annotated[int, "Maximum number of questions to return"] = 3
) -> str:
    """Search Stack Overflow for programming questions and answers."""
    logger.info(f"Stack Overflow searching for: '{query}'")
    
    try:
        # Using Stack Exchange API
        url = "https://api.stackexchange.com/2.3/search/advanced"
        params = {
            "order": "desc",
            "sort": "relevance",
            "q": query,
            "site": "stackoverflow",
            "pagesize": max_results,
            "filter": "withbody"
        }
        
        response = requests.get(url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        if not data.get("items"):
            return "No Stack Overflow questions found for this query."
        
        formatted_results = []
        for question in data["items"]:
            title = question["title"]
            body = question["body"][:300] + "..." if len(question["body"]) > 300 else question["body"]
            score = question["score"]
            answers = question["answer_count"]
            
            # Clean HTML tags from body
            import re
            body = re.sub(r'<[^>]+>', '', body)
            
            formatted_results.append(
                f"Question: {title}\n"
                f"Score: {score} | Answers: {answers}\n"
                f"Preview: {body}\n"
                f"URL: {question['link']}\n"
                f"References: 💡 [Stack Overflow]({question['link']}) | 🌐 {TOOL_REFERENCES['stackoverflow']['url']}"
            )
        
        logger.info("Stack Overflow search complete.")
        return "\n\n---\n\n".join(formatted_results)
        
    except Exception as e:
        logger.error(f"Stack Overflow search failed: {e}")
        return f"Stack Overflow search failed. Error: {str(e)}"
