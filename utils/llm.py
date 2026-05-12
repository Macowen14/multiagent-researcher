import os
import logging
from langchain_core.language_models.chat_models import BaseChatModel

logger = logging.getLogger(__name__)


class LLMFactory:
    """
    A factory class to initialize and return the appropriate LLM
    based on the AI_PROVIDER environment variable.
    """

    @staticmethod
    def get_llm(model_name: str = None, temperature: float = 0.0) -> BaseChatModel:
        """
        Checks the AI_PROVIDER environment variable and returns the configured LLM.

        Args:
            model_name (str, optional): The specific model to use. If not provided,
                                        defaults will be chosen based on the provider.
            temperature (float): The temperature setting for the model. Defaults to 0.0.

        Returns:
            BaseChatModel: An initialized LangChain chat model.
        """
        provider = os.getenv("AI_PROVIDER", "openai").strip().lower()

        if provider == "openai":
            from langchain_openai import ChatOpenAI

            # Default to gpt-4o if no model is explicitly passed or set in environment
            model = model_name or os.getenv("OPENAI_MODEL", "gpt-4o")
            logger.info(f"Initializing OpenAI LLM with model: {model}")

            return ChatOpenAI(model=model, temperature=temperature)

        elif provider == "ollama":
            from langchain_ollama import ChatOllama

            # Default to llama3 if no model is explicitly passed or set in environment
            model = model_name or os.getenv("OLLAMA_MODEL", "llama3")
            logger.info(f"Initializing Ollama LLM with model: {model}")

            # Note: Ensure you have your local Ollama instance running
            return ChatOllama(model=model, temperature=temperature)

        else:
            raise ValueError(
                f"Unsupported AI_PROVIDER: '{provider}'. "
                "Please set AI_PROVIDER to either 'openai' or 'ollama'."
            )
