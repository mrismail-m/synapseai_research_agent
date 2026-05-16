"""
Search tools module for executing web queries.

Provides a configured Tavily search instance for the research agent.
"""

import os
from langchain_community.tools.tavily_search import TavilySearchResults
from src.config import config

# Ensure the TAVILY_API_KEY is in the environment for the tool
os.environ["TAVILY_API_KEY"] = config.TAVILY_API_KEY or ""

def get_tavily_tool(max_results: int = config.MAX_SEARCH_RESULTS) -> TavilySearchResults:
    """
    Returns a configured Tavily search tool instance.
    
    Args:
        max_results (int): The maximum number of search results to return.
            Defaults to the MAX_SEARCH_RESULTS defined in config.
            
    Returns:
        TavilySearchResults: The initialized search tool.
    """
    return TavilySearchResults(max_results=max_results)
