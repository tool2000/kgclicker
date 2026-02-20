from dotenv import load_dotenv
from src.kg_gen import KGGen
import os
import pytest

load_dotenv()


def _get_test_llm_config():
    """Build LLM config from env vars, with Azure fallback."""
    # Check for explicit LLM_MODEL first
    model = os.getenv("LLM_MODEL")
    api_key = os.getenv("LLM_API_KEY")
    api_base = os.getenv("LLM_API_BASE")
    api_version = os.getenv("LLM_API_VERSION")

    # Fall back to Azure config if no explicit model set
    if not model:
        azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
        azure_deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT")
        azure_key = os.getenv("AZURE_OPENAI_API_KEY")
        if azure_endpoint and azure_deployment and azure_key:
            model = f"azure/{azure_deployment}"
            api_key = azure_key
            api_base = azure_endpoint.rstrip("/")
            api_version = os.getenv("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")

    return {
        "model": model or "openai/gpt-5-nano",
        "api_key": api_key,
        "api_base": api_base,
        "api_version": api_version,
    }


@pytest.fixture
def kg():
    cfg = _get_test_llm_config()
    return KGGen(
        model=cfg["model"],
        api_key=cfg["api_key"],
        api_base=cfg["api_base"],
        api_version=cfg["api_version"],
        temperature=float(os.getenv("LLM_TEMPERATURE", "1.0")),
        retrieval_model=os.getenv("RETRIEVAL_MODEL", "all-MiniLM-L6-v2"),
    )
