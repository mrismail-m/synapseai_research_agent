"""
Configuration module for the Synapse Research Assistant.

Loads environment variables, validates required keys, and defines 
system-wide constants to avoid hardcoded values throughout the codebase.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

class Config:
    """
    Configuration class to manage environment variables and system constants.
    
    Attributes:
        GROQ_API_KEY (str): API key for the Groq LLM service.
        TAVILY_API_KEY (str): API key for the Tavily search service.
        LLM_MODEL_NAME (str): Default LLM model identifier.
        LLM_TEMPERATURE_STRICT (float): Temperature for deterministic agents (e.g., routing, validation).
        LLM_TEMPERATURE_CREATIVE (float): Temperature for generative agents (e.g., synthesis).
        MAX_RESEARCH_ATTEMPTS (int): Maximum loops allowed for the research/validation cycle.
        CONFIDENCE_THRESHOLD (int): Minimum confidence score required to skip validation.
        MAX_SEARCH_RESULTS (int): Number of results returned per Tavily search query.
    """
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")
    
    # Model configuration
    LLM_MODEL_NAME = "llama-3.1-8b-instant"
    LLM_TEMPERATURE_STRICT = 0.0
    LLM_TEMPERATURE_CREATIVE = 0.1
    
    # Graph execution constants
    MAX_RESEARCH_ATTEMPTS = 3
    CONFIDENCE_THRESHOLD = 6
    
    # Tool constants
    MAX_SEARCH_RESULTS = 5

    @classmethod
    def validate(cls) -> None:
        """
        Validates that all required configuration keys are present.
        
        Raises:
            ValueError: If any essential environment variable is missing.
        """
        missing = []
        if not cls.GROQ_API_KEY:
            missing.append("GROQ_API_KEY")
        if not cls.TAVILY_API_KEY:
            missing.append("TAVILY_API_KEY")
        
        if missing:
            raise ValueError(f"Missing environment variables: {', '.join(missing)}")

# Global config instance
config = Config()
