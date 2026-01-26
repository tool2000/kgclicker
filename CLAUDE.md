# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

kg-gen is a Python library for extracting knowledge graphs from text using LLMs. It processes text through a 3-step pipeline (entity extraction → relation extraction → deduplication) and supports multiple LLM providers via LiteLLM.

## Common Commands

```bash
# Install in development mode
pip install -e '.[dev]'

# Install with MCP support
pip install -e '.[mcp,dev]'

# Run all tests
pytest

# Run specific test file
pytest tests/test_basic.py -v

# Run a single test
pytest tests/test_basic.py::test_basic -v

# Run MCP tests
pytest mcp/tests/ -v

# Lint and format (via pre-commit)
ruff check .
ruff format .

# Start MCP server
kggen mcp --keep-memory --storage-path ./memory.json
```

## Environment Variables for Testing

Tests require LLM API access. Configure via `.env` or environment:
- `LLM_MODEL` - Model to use (default: `openai/gpt-5-nano`)
- `LLM_API_KEY` - API key for the model
- `LLM_TEMPERATURE` - Temperature (default: `1.0` for gpt-5 family)
- `RETRIEVAL_MODEL` - Embedding model (default: `all-MiniLM-L6-v2`)

Note: gpt-5 family models require `temperature >= 1.0` and `max_tokens >= 16000`.

## Architecture

### Core Pipeline (src/kg_gen/steps/)

1. **`_1_get_entities.py`** - Extracts entities from text using DSPy or LiteLLM
2. **`_2_get_relations.py`** - Extracts (subject, predicate, object) triples
3. **`_3_deduplicate.py`** - Deduplicates using three methods:
   - `SEMHASH` - Fast semantic hashing (similarity threshold 0.95)
   - `LM_BASED` - KNN clustering + LLM-based intra-cluster dedup
   - `FULL` - Combined semhash then LM-based

### Key Data Structure (src/kg_gen/models.py)

```python
class Graph(BaseModel):
    entities: set[str]                           # Extracted entities
    edges: set[str]                              # Relation predicates
    relations: set[Tuple[str, str, str]]         # (subject, predicate, object)
    entity_clusters: Optional[dict[str, set[str]]]  # Dedup clusters
    edge_clusters: Optional[dict[str, set[str]]]
```

### Main Entry Points

- **`KGGen` class** (`src/kg_gen/kg_gen.py`) - Main API: `generate()`, `deduplicate()`, `aggregate()`, `visualize()`
- **CLI** (`src/kg_gen/cli.py`) - `kggen mcp` command
- **MCP Server** (`mcp/server.py`) - Memory tools for AI agents

### Model Provider Format

Uses LiteLLM routing: `{provider}/{model_name}`
- `openai/gpt-4o`
- `gemini/gemini-2.5-flash`
- `ollama_chat/deepseek-r1:14b`
- `anthropic/claude-3-5-sonnet-20241022`

Custom endpoints via `api_base` parameter.

## Key Utilities (src/kg_gen/utils/)

- `chunk_text.py` - Sentence-boundary-aware text chunking
- `deduplicate.py` - Semantic hashing with normalization/singularization
- `llm_deduplicate.py` - Embedding-based KNN clustering
- `visualize_kg.py` - Interactive HTML visualization
- `neo4j_integration.py` - Neo4j database upload
- `vector_store.py` - Vector storage for retrieval
- `document_parser.py` - PDF/DOCX/PPTX parsing

## Testing Patterns

Tests use pytest fixtures from `tests/fixtures.py`:
```python
@pytest.fixture
def kg():
    return KGGen(
        model=os.getenv("LLM_MODEL", "openai/gpt-5-nano"),
        api_key=os.getenv("LLM_API_KEY"),
        temperature=float(os.getenv("LLM_TEMPERATURE", "1.0")),
        retrieval_model=os.getenv("RETRIEVAL_MODEL", "all-MiniLM-L6-v2"),
    )
```

Tests use fuzzy subset matching due to non-deterministic LLM outputs at temperature 1.0.
