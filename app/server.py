from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List
import json
import os
import re
import uuid

import logging
from fastapi import FastAPI, File, Form, HTTPException, UploadFile, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ValidationError
import litellm

litellm.drop_params = True

from kg_gen import KGGen
from kg_gen.models import Graph
from kg_gen.utils.visualize_kg import _build_view_model
from kg_gen.utils.document_parser import (
    extract_text,
    is_supported_format,
    SUPPORTED_EXTENSIONS,
    DocumentParseError,
)
from kg_gen.utils.graph_storage import GraphStorage, get_storage
from kg_gen.utils.vector_store import VectorStore
from kg_gen.utils.graph_analysis import (
    compute_basic_stats,
    compute_centrality,
    detect_communities,
    compute_connected_components,
    find_shortest_path,
    find_all_paths,
    get_entity_neighbors,
    compute_full_analysis,
)

APP_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = (
    APP_DIR.parent / "src" / "kg_gen" / "utils" / "template.html"
).resolve()
DATA_ROOT = (APP_DIR.parent / "app" / "examples").resolve()
STORAGE_DIR = APP_DIR.parent / "data" / "graphs"
VECTOR_DIR = APP_DIR.parent / "data" / "vectors"


@dataclass(frozen=True)
class ExampleGraph:
    slug: str
    title: str
    path: Path
    wiki_url: str


# TODO: this will be read from huggingface once it is uploaded.
EXAMPLE_GRAPHS: tuple[ExampleGraph, ...] = ()
for file in DATA_ROOT.glob("*.json"):
    EXAMPLE_GRAPHS += (
        ExampleGraph(
            slug=file.stem,
            title=file.stem,
            path=file,
            wiki_url=f"https://en.wikipedia.org/wiki/{file.stem}",
        ),
    )


EXAMPLE_INDEX = {
    example.slug: example for example in EXAMPLE_GRAPHS if example.path.exists()
}

if len(EXAMPLE_INDEX) < len(EXAMPLE_GRAPHS):
    missing = [
        example.slug for example in EXAMPLE_GRAPHS if example.slug not in EXAMPLE_INDEX
    ]
    logger = logging.getLogger("kg_gen_app")
    logger.warning("Example graphs missing on disk: %s", ", ".join(missing))

if not TEMPLATE_PATH.exists():
    raise RuntimeError(f"Template not found at {TEMPLATE_PATH}")

logger = logging.getLogger("kg_gen_app")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


def _get_azure_config() -> dict:
    """Get Azure OpenAI configuration from environment variables."""
    endpoint = os.environ.get("AZURE_OPENAI_ENDPOINT")
    deployment = os.environ.get("AZURE_OPENAI_DEPLOYMENT")
    api_key = os.environ.get("AZURE_OPENAI_API_KEY")
    api_version = os.environ.get("AZURE_OPENAI_API_VERSION", "2025-03-01-preview")

    if endpoint and deployment and api_key:
        normalized_endpoint = endpoint.rstrip('/')
        if "/openai/deployments/" in normalized_endpoint:
            normalized_endpoint = normalized_endpoint.split("/openai/deployments/")[0]
        base_url = normalized_endpoint
        return {
            "api_key": api_key,
            "api_base": base_url,
            "model": f"azure/{deployment}",
            "api_version": api_version,
        }
    return {}


app = FastAPI(title="kg-gen explorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize KGGen with Azure OpenAI if env vars are set
_init_azure = _get_azure_config()
if _init_azure:
    kg_gen = KGGen(
        model=_init_azure["model"],
        api_key=_init_azure["api_key"],
        api_base=_init_azure["api_base"],
        api_version=_init_azure.get("api_version"),
    )
else:
    kg_gen = KGGen()

# Initialize graph storage
storage = get_storage(STORAGE_DIR)
vector_store = VectorStore(VECTOR_DIR)


# =============================================================================
# Request/Response Models
# =============================================================================


class GraphSaveRequest(BaseModel):
    graph_id: str
    name: Optional[str] = None
    description: Optional[str] = None
    source_documents: Optional[List[str]] = None


class GraphMergeRequest(BaseModel):
    target_graph_id: str
    source_graph_ids: List[str]
    new_name: Optional[str] = None


class VectorIndexRequest(BaseModel):
    text: str
    chunk_size: Optional[int] = 800
    embedding_model: Optional[str] = None


class RagQueryRequest(BaseModel):
    graph_id: str
    question: str
    top_k_nodes: Optional[int] = 8
    top_k_chunks: Optional[int] = 6
    retrieval_model: Optional[str] = None
    model: Optional[str] = None
    api_key: Optional[str] = None
    temperature: Optional[float] = None


class PathFindRequest(BaseModel):
    source: str
    target: str
    max_length: Optional[int] = 5
    find_all: Optional[bool] = False


class NeighborRequest(BaseModel):
    entity: str
    depth: Optional[int] = 1


# =============================================================================
# Helper Functions
# =============================================================================


def _clean_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in {"true", "1", "yes", "on"}


def _extract_litellm_text(response: object) -> str:
    content = getattr(response, "output", None)
    if content and isinstance(content, list):
        last = content[-1]
        last_content = getattr(last, "content", None)
        if last_content and isinstance(last_content, list):
            text_part = last_content[0]
            text = getattr(text_part, "text", None)
            if text:
                return text
    return str(response)


def _sanitize_graph_id(name: str) -> str:
    """Convert a name to a safe graph ID."""
    # Remove or replace unsafe characters
    safe = re.sub(r"[^\w\-]", "_", name.lower())
    # Remove multiple underscores
    safe = re.sub(r"_+", "_", safe)
    return safe.strip("_")[:64]  # Limit length


# =============================================================================
# Original Endpoints
# =============================================================================


@app.get("/", response_class=HTMLResponse)
async def serve_index() -> HTMLResponse:
    index_path = APP_DIR / "index.html"
    if not index_path.exists():
        raise HTTPException(status_code=500, detail="index.html missing")
    logger.debug("Serving index page")
    return HTMLResponse(index_path.read_text(encoding="utf-8"))


@app.get("/template")
async def serve_template() -> FileResponse:
    logger.debug("Serving visualization template from %s", TEMPLATE_PATH)
    return FileResponse(TEMPLATE_PATH, media_type="text/html")


@app.get("/api/examples")
async def list_examples() -> JSONResponse:
    logger.debug("Listing built-in example graphs")
    items = [
        {"slug": example.slug, "title": example.title, "wiki_url": example.wiki_url}
        for example in sorted(
            EXAMPLE_INDEX.values(), key=lambda item: item.title.lower()
        )
    ]
    return JSONResponse(items)


@app.get("/api/examples/{slug}")
async def load_example(slug: str) -> JSONResponse:
    example = EXAMPLE_INDEX.get(slug)
    if example is None:
        raise HTTPException(status_code=404, detail=f"Example '{slug}' not found")

    if not example.path.exists():
        logger.error("Example graph missing: %s", example.path)
        raise HTTPException(status_code=404, detail=f"Example '{slug}' is unavailable")

    try:
        payload = json.loads(example.path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        logger.exception("Failed to parse example graph %s", slug)
        raise HTTPException(
            status_code=500, detail=f"Example '{slug}' is invalid: {exc}"
        )

    logger.info("Loaded example graph '%s' from %s", slug, example.path)
    return JSONResponse(payload)


@app.post("/api/graph/view")
async def build_view(graph: Graph) -> JSONResponse:
    """Convert a raw KGGen graph payload into the template view model."""
    logger.info(
        "Received request to build view: entities=%s relations=%s",
        len(graph.entities),
        len(graph.relations),
    )
    try:
        view = _build_view_model(graph)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to build view model")
        raise HTTPException(status_code=500, detail=f"Failed to build view: {exc}")
    logger.info(
        "View model ready: nodes=%s edges=%s", len(view["nodes"]), len(view["edges"])
    )
    return JSONResponse({"view": view, "graph": graph.model_dump(mode="json")})


@app.post("/api/generate")
async def generate_graph(
    api_key: Optional[str] = Form(None, description="OpenAI API key (optional if Azure configured)"),
    model: str = Form("openai/gpt-4o"),
    context: Optional[str] = Form(None),
    chunk_size: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),
    cluster: Optional[str] = Form(None),
    source_text: Optional[str] = Form(None),
    text_file: Optional[UploadFile] = File(None),
    retrieval_model: Optional[str] = Form("sentence-transformers/all-mpnet-base-v2"),
    graph_id: Optional[str] = Form(None, description="Save graph with this ID"),
    merge_with: Optional[str] = Form(None, description="Merge with existing graph ID"),
    build_vector_index: Optional[str] = Form("true", description="Build vector index for RAG"),
) -> JSONResponse:
    """Generate a knowledge graph from text or documents."""
    text_fragments: list[str] = []
    source_documents: list[str] = []

    # Check for Azure OpenAI configuration
    azure_config = _get_azure_config()
    effective_api_key = api_key or azure_config.get("api_key")
    effective_model = azure_config.get("model") if azure_config else model
    effective_api_base = azure_config.get("api_base")

    if not effective_api_key:
        raise HTTPException(
            status_code=400,
            detail="API key required. Provide api_key or set AZURE_OPENAI_* environment variables.",
        )

    cleaned_text = _clean_str(source_text)
    if cleaned_text:
        text_fragments.append(cleaned_text)

    if text_file is not None:
        filename = text_file.filename or "unknown"
        source_documents.append(filename)

        try:
            contents = await text_file.read()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to read uploaded file")
            raise HTTPException(
                status_code=400, detail=f"Reading file failed: {exc}"
            )

        # Check if it's a supported document format
        if is_supported_format(filename):
            try:
                extracted_text = extract_text(contents, filename)
                text_fragments.append(extracted_text)
                logger.info(f"Extracted {len(extracted_text)} chars from {filename}")
            except DocumentParseError as exc:
                logger.exception("Failed to parse document")
                raise HTTPException(status_code=400, detail=str(exc))
        else:
            # Assume plain text
            try:
                decoded = contents.decode("utf-8")
            except UnicodeDecodeError as exc:
                logger.exception("Uploaded text file must be UTF-8")
                raise HTTPException(
                    status_code=400, detail=f"Text file must be UTF-8: {exc}"
                )
            cleaned_file_text = _clean_str(decoded)
            if cleaned_file_text:
                text_fragments.append(cleaned_file_text)

    if not text_fragments:
        raise HTTPException(
            status_code=400, detail="Provide inline text or upload a file"
        )

    request_text = "\n\n".join(text_fragments)

    numeric_chunk: Optional[int] = None
    if chunk_size:
        try:
            numeric_chunk = int(chunk_size)
        except ValueError as exc:
            logger.warning("Invalid chunk_size received: %s", chunk_size)
            raise HTTPException(
                status_code=400, detail=f"chunk_size must be an integer: {exc}"
            )
        if numeric_chunk <= 0:
            numeric_chunk = None

    numeric_temperature: Optional[float] = None
    if temperature:
        try:
            numeric_temperature = float(temperature)
        except ValueError as exc:
            logger.warning("Invalid temperature received: %s", temperature)
            raise HTTPException(
                status_code=400, detail=f"temperature must be numeric: {exc}"
            )

    # Validate temperature for gpt-5 reasoning models (not chat/5.2+ variants)
    _is_gpt5_reasoning = (
        effective_model
        and "gpt-5" in effective_model
        and "gpt-5.2" not in effective_model
        and "gpt-5-chat" not in effective_model
    )
    if _is_gpt5_reasoning:
        if numeric_temperature is not None and numeric_temperature < 1.0:
            raise HTTPException(
                status_code=400,
                detail="Temperature must be 1.0 or higher for gpt-5 reasoning models",
            )
        if numeric_temperature is None:
            numeric_temperature = 1.0

    kg_gen.init_model(
        model=effective_model,
        api_key=effective_api_key,
        api_base=effective_api_base,
        temperature=numeric_temperature,
        retrieval_model=retrieval_model,
    )

    logger.info(
        "Generating graph via KGGen: model=%s chunk_size=%s context_len=%s text_len=%s temperature=%s retrieval_model=%s",
        effective_model,
        numeric_chunk,
        len((_clean_str(context) or "")),
        len(request_text),
        numeric_temperature,
        retrieval_model,
    )
    try:
        graph = kg_gen.generate(
            input_data=request_text,
            model=effective_model,
            api_key=effective_api_key,
            api_base=effective_api_base,
            context=_clean_str(context) or "",
            chunk_size=numeric_chunk,
            temperature=numeric_temperature,
        )
    except ValidationError as exc:
        logger.exception("KGGen returned validation error")
        raise HTTPException(status_code=400, detail=f"Invalid graph result: {exc}")
    except Exception as exc:
        logger.exception("KGGen generation failed")
        raise HTTPException(status_code=500, detail=f"KGGen failed: {exc}")

    # Merge with existing graph if requested
    if merge_with:
        existing_graph = storage.load(merge_with)
        if existing_graph:
            logger.info(f"Merging with existing graph '{merge_with}'")
            graph = kg_gen.aggregate([existing_graph, graph])
            source_documents = list(set(
                (storage.get_metadata(merge_with) or {}).get("source_documents", []) +
                source_documents
            ))

    # Auto-generate graph_id if vector index requested but no id given
    want_vector = _parse_bool(build_vector_index)
    if not graph_id and want_vector:
        graph_id = f"graph-{uuid.uuid4().hex[:8]}"

    # Save graph if graph_id provided (or auto-generated)
    saved_graph_id = None
    if graph_id:
        sanitized_id = _sanitize_graph_id(graph_id)
        saved_graph_id = sanitized_id
        storage.save(
            graph,
            graph_id=sanitized_id,
            name=graph_id,
            source_documents=source_documents,
        )
        logger.info(f"Saved graph as '{sanitized_id}'")

        if want_vector:
            try:
                vector_meta = vector_store.build_index(
                    sanitized_id,
                    request_text,
                    retrieval_model,
                    numeric_chunk or 800,
                )
                storage.update_vector_index(sanitized_id, vector_meta.__dict__)
                logger.info(f"Vector index built for '{sanitized_id}'")
            except Exception as exc:
                logger.warning("Vector index build failed for %s: %s", sanitized_id, exc)

    try:
        view = _build_view_model(graph)
    except Exception as exc:  # pragma: no cover - defensive
        logger.exception("Failed to build view model after generation")
        raise HTTPException(status_code=500, detail=f"Failed to build view: {exc}")

    logger.info(
        "Graph generation complete: entities=%s relations=%s",
        len(graph.entities),
        len(graph.relations),
    )
    result = {"view": view, "graph": graph.model_dump(mode="json")}
    if saved_graph_id:
        result["graph_id"] = saved_graph_id
    return JSONResponse(result)


# =============================================================================
# Document Upload Endpoints
# =============================================================================


@app.get("/api/documents/formats")
async def get_supported_formats() -> JSONResponse:
    """Get list of supported document formats."""
    return JSONResponse({
        "formats": sorted(list(SUPPORTED_EXTENSIONS)),
        "description": {
            ".pdf": "PDF documents",
            ".docx": "Microsoft Word (DOCX)",
            ".doc": "Microsoft Word (DOC) - limited support",
            ".pptx": "Microsoft PowerPoint (PPTX)",
            ".ppt": "Microsoft PowerPoint (PPT) - limited support",
            ".txt": "Plain text files",
        }
    })


@app.post("/api/documents/extract")
async def extract_document_text(
    file: UploadFile = File(..., description="Document file to extract text from"),
) -> JSONResponse:
    """Extract text from an uploaded document without generating a graph."""
    filename = file.filename or "unknown"

    if not is_supported_format(filename):
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file format. Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}",
        )

    try:
        contents = await file.read()
        text = extract_text(contents, filename)
    except DocumentParseError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("Document extraction failed")
        raise HTTPException(status_code=500, detail=f"Extraction failed: {exc}")

    return JSONResponse({
        "filename": filename,
        "text": text,
        "character_count": len(text),
        "word_count": len(text.split()),
    })


@app.post("/api/documents/upload-multiple")
async def upload_multiple_documents(
    files: List[UploadFile] = File(..., description="Multiple documents to process"),
    api_key: Optional[str] = Form(None),
    model: str = Form("openai/gpt-4o"),
    context: Optional[str] = Form(None),
    chunk_size: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),
    cluster: Optional[str] = Form(None),
    retrieval_model: Optional[str] = Form("sentence-transformers/all-mpnet-base-v2"),
    graph_id: Optional[str] = Form(None),
    merge_with: Optional[str] = Form(None),
    build_vector_index: Optional[str] = Form("true"),
) -> JSONResponse:
    """Upload multiple documents and generate a combined knowledge graph."""
    if not files:
        raise HTTPException(status_code=400, detail="No files uploaded")

    # Check for Azure OpenAI configuration
    azure_config = _get_azure_config()
    effective_api_key = api_key or azure_config.get("api_key")
    effective_model = azure_config.get("model") if azure_config else model
    effective_api_base = azure_config.get("api_base")

    if not effective_api_key:
        raise HTTPException(
            status_code=400,
            detail="API key required. Provide api_key or set AZURE_OPENAI_* environment variables.",
        )

    text_fragments = []
    source_documents = []
    errors = []

    for file in files:
        filename = file.filename or "unknown"
        source_documents.append(filename)

        try:
            contents = await file.read()

            if is_supported_format(filename):
                text = extract_text(contents, filename)
            else:
                text = contents.decode("utf-8")

            if text.strip():
                text_fragments.append(f"=== Document: {filename} ===\n{text}")
                logger.info(f"Extracted {len(text)} chars from {filename}")

        except Exception as exc:
            errors.append(f"{filename}: {str(exc)}")
            logger.warning(f"Failed to process {filename}: {exc}")

    if not text_fragments:
        raise HTTPException(
            status_code=400,
            detail=f"No text could be extracted. Errors: {'; '.join(errors)}"
        )

    combined_text = "\n\n".join(text_fragments)

    numeric_temperature = None
    if temperature:
        try:
            numeric_temperature = float(temperature)
        except ValueError:
            pass

    numeric_chunk = None
    if chunk_size:
        try:
            numeric_chunk = int(chunk_size)
        except ValueError:
            pass

    kg_gen.init_model(
        model=effective_model,
        api_key=effective_api_key,
        api_base=effective_api_base,
        temperature=numeric_temperature,
        retrieval_model=retrieval_model,
    )

    try:
        graph = kg_gen.generate(
            input_data=combined_text,
            context=_clean_str(context) or "",
            chunk_size=numeric_chunk or 8000,
            temperature=numeric_temperature,
        )
    except Exception as exc:
        logger.exception("Graph generation failed")
        raise HTTPException(status_code=500, detail=f"Generation failed: {exc}")

    # Merge with existing graph if requested
    if merge_with:
        existing_graph = storage.load(merge_with)
        if existing_graph:
            graph = kg_gen.aggregate([existing_graph, graph])
            source_documents = list(set(
                (storage.get_metadata(merge_with) or {}).get("source_documents", []) +
                source_documents
            ))

    # Auto-generate graph_id if vector index requested but no id given
    want_vector = _parse_bool(build_vector_index)
    if not graph_id and want_vector:
        graph_id = f"graph-{uuid.uuid4().hex[:8]}"

    # Save graph if graph_id provided (or auto-generated)
    saved_graph_id = None
    if graph_id:
        sanitized_id = _sanitize_graph_id(graph_id)
        saved_graph_id = sanitized_id
        storage.save(
            graph,
            graph_id=sanitized_id,
            name=graph_id,
            source_documents=source_documents,
        )

        if want_vector:
            try:
                vector_meta = vector_store.build_index(
                    sanitized_id,
                    combined_text,
                    retrieval_model,
                    numeric_chunk or 800,
                )
                storage.update_vector_index(sanitized_id, vector_meta.__dict__)
                logger.info(f"Vector index built for '{sanitized_id}'")
            except Exception as exc:
                logger.warning("Vector index build failed for %s: %s", sanitized_id, exc)

    view = _build_view_model(graph)

    result = {
        "view": view,
        "graph": graph.model_dump(mode="json"),
        "processed_files": source_documents,
        "errors": errors if errors else None,
    }
    if saved_graph_id:
        result["graph_id"] = saved_graph_id
    return JSONResponse(result)


# =============================================================================
# Graph Storage Endpoints
# =============================================================================


@app.get("/api/graphs")
async def list_graphs() -> JSONResponse:
    """List all saved graphs."""
    graphs = storage.list_graphs()
    return JSONResponse({"graphs": graphs})


@app.get("/api/graphs/{graph_id}")
async def get_graph(graph_id: str, version: Optional[str] = None) -> JSONResponse:
    """Load a saved graph."""
    graph = storage.load(graph_id, version)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")

    metadata = storage.get_metadata(graph_id)
    view = _build_view_model(graph)

    return JSONResponse({
        "view": view,
        "graph": graph.model_dump(mode="json"),
        "metadata": metadata,
    })


@app.post("/api/graphs/{graph_id}")
async def save_graph(
    graph_id: str,
    graph: Graph,
    name: Optional[str] = Query(None),
    description: Optional[str] = Query(None),
) -> JSONResponse:
    """Save a graph."""
    metadata = storage.save(
        graph,
        graph_id=_sanitize_graph_id(graph_id),
        name=name or graph_id,
        description=description,
    )
    return JSONResponse({"status": "saved", "metadata": metadata})


@app.post("/api/graphs/{graph_id}/vector-index")
async def build_vector_index(graph_id: str, request: VectorIndexRequest) -> JSONResponse:
    sanitized_id = _sanitize_graph_id(graph_id)
    if not storage.exists(sanitized_id):
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")

    embedding_model = request.embedding_model or "sentence-transformers/all-mpnet-base-v2"
    chunk_size = request.chunk_size or 800

    try:
        meta = vector_store.build_index(
            sanitized_id,
            request.text,
            embedding_model,
            chunk_size,
        )
    except Exception as exc:
        logger.exception("Vector index build failed")
        raise HTTPException(status_code=400, detail=f"Vector index failed: {exc}")

    storage.update_vector_index(sanitized_id, meta.__dict__)
    return JSONResponse({"status": "indexed", "vector_index": meta.__dict__})


@app.delete("/api/graphs/{graph_id}")
async def delete_graph(
    graph_id: str,
    delete_history: bool = Query(False),
) -> JSONResponse:
    """Delete a saved graph."""
    if not storage.delete(graph_id, delete_history):
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")
    return JSONResponse({"status": "deleted", "graph_id": graph_id})


@app.get("/api/graphs/{graph_id}/versions")
async def get_graph_versions(graph_id: str) -> JSONResponse:
    """Get version history for a graph."""
    if not storage.exists(graph_id):
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")

    versions = storage.get_versions(graph_id)
    return JSONResponse({"graph_id": graph_id, "versions": versions})


@app.post("/api/graphs/merge")
async def merge_graphs(request: GraphMergeRequest) -> JSONResponse:
    """Merge multiple graphs into one."""
    graphs_to_merge = []
    source_docs = []

    for source_id in request.source_graph_ids:
        graph = storage.load(source_id)
        if graph is None:
            raise HTTPException(
                status_code=404, detail=f"Source graph '{source_id}' not found"
            )
        graphs_to_merge.append(graph)
        meta = storage.get_metadata(source_id)
        if meta:
            source_docs.extend(meta.get("source_documents", []))

    if not graphs_to_merge:
        raise HTTPException(status_code=400, detail="No valid source graphs provided")

    # Load target graph if it exists
    target_graph = storage.load(request.target_graph_id)
    if target_graph:
        graphs_to_merge.insert(0, target_graph)
        target_meta = storage.get_metadata(request.target_graph_id)
        if target_meta:
            source_docs = target_meta.get("source_documents", []) + source_docs

    merged = kg_gen.aggregate(graphs_to_merge)

    metadata = storage.save(
        merged,
        graph_id=_sanitize_graph_id(request.target_graph_id),
        name=request.new_name or request.target_graph_id,
        source_documents=list(set(source_docs)),
    )

    view = _build_view_model(merged)

    return JSONResponse({
        "view": view,
        "graph": merged.model_dump(mode="json"),
        "metadata": metadata,
    })


@app.post("/api/rag/query")
async def rag_query(request: RagQueryRequest) -> JSONResponse:
    sanitized_id = _sanitize_graph_id(request.graph_id)
    graph = storage.load(sanitized_id)
    has_graph = graph is not None
    has_vector_index = vector_store.load_index(sanitized_id) is not None

    if not has_graph and not has_vector_index:
        raise HTTPException(
            status_code=404,
            detail=f"Graph '{request.graph_id}' not found and no vector index exists",
        )

    azure_config = _get_azure_config()
    effective_api_key = request.api_key or azure_config.get("api_key")
    effective_model = request.model or azure_config.get("model") or "openai/gpt-4o"
    effective_api_base = azure_config.get("api_base")
    api_version = azure_config.get("api_version")

    if not effective_api_key:
        raise HTTPException(
            status_code=400,
            detail="API key required. Provide api_key or set AZURE_OPENAI_* environment variables.",
        )

    retrieval_model = request.retrieval_model or "sentence-transformers/all-mpnet-base-v2"
    kg_gen.init_model(
        model=effective_model,
        api_key=effective_api_key,
        api_base=effective_api_base,
        temperature=request.temperature or 0.2,
        retrieval_model=retrieval_model,
    )

    top_nodes = []
    kg_context = ""
    if has_graph:
        try:
            node_embeddings, _ = kg_gen.generate_embeddings(graph, model=kg_gen.retrieval_model)
            nx_graph = KGGen.to_nx(graph)
            top_nodes, _, kg_context = kg_gen.retrieve(
                request.question,
                node_embeddings,
                nx_graph,
                model=kg_gen.retrieval_model,
                k=request.top_k_nodes or 8,
            )
        except Exception as exc:
            logger.warning("KG retrieval failed: %s", exc)

    vector_results = []
    if has_vector_index:
        try:
            vector_results = vector_store.query(
                sanitized_id,
                request.question,
                request.top_k_chunks or 6,
                embedding_model=retrieval_model,
            )
        except Exception as exc:
            logger.warning("Vector search failed: %s", exc)

    if not kg_context and not vector_results:
        raise HTTPException(
            status_code=400,
            detail="No context found from either KG or vector index",
        )

    vector_context = "\n".join(
        [f"- {chunk['text']}" for chunk in vector_results if chunk.get("text")]
    )

    prompt_parts = [
        "Answer the question using the available context. "
        "If the answer is not supported, say you do not know.\n\n"
        f"Question:\n{request.question}\n"
    ]
    if kg_context:
        prompt_parts.append(f"\nKnowledge Graph Context:\n{kg_context}\n")
    if vector_context:
        prompt_parts.append(f"\nDocument Excerpts:\n{vector_context}\n")
    prompt = "".join(prompt_parts)

    litellm_kwargs = {
        "model": effective_model,
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": prompt},
        ],
        "api_key": effective_api_key,
        "api_base": effective_api_base,
        "temperature": request.temperature or 0.2,
    }
    if api_version:
        litellm_kwargs["api_version"] = api_version

    try:
        response = litellm.completion(**litellm_kwargs)
        answer = response.choices[0].message.content
    except Exception as exc:
        logger.exception("RAG answer generation failed")
        raise HTTPException(status_code=500, detail=f"Answer generation failed: {exc}")

    return JSONResponse({
        "answer": answer,
        "question": request.question,
        "kg_context": kg_context or None,
        "kg_top_nodes": [
            {"node": node, "score": score}
            for node, score in top_nodes
        ] if top_nodes else None,
        "vector_chunks": vector_results if vector_results else None,
        "sources_used": {
            "knowledge_graph": has_graph and bool(kg_context),
            "vector_index": has_vector_index and bool(vector_results),
        },
    })


# =============================================================================
# Graph Analysis Endpoints
# =============================================================================


@app.post("/api/analysis/stats")
async def analyze_stats(graph: Graph) -> JSONResponse:
    """Compute basic statistics for a graph."""
    stats = compute_basic_stats(graph)
    return JSONResponse(stats)


@app.post("/api/analysis/centrality")
async def analyze_centrality(
    graph: Graph,
    top_k: int = Query(20, description="Number of top entities to return"),
) -> JSONResponse:
    """Compute centrality metrics for a graph."""
    centrality = compute_centrality(graph, top_k)
    return JSONResponse(centrality)


@app.post("/api/analysis/communities")
async def analyze_communities(
    graph: Graph,
    method: str = Query("louvain", description="Community detection method"),
) -> JSONResponse:
    """Detect communities in a graph."""
    communities = detect_communities(graph, method)
    return JSONResponse(communities)


@app.post("/api/analysis/components")
async def analyze_components(graph: Graph) -> JSONResponse:
    """Find connected components in a graph."""
    components = compute_connected_components(graph)
    return JSONResponse(components)


@app.post("/api/analysis/path")
async def find_path(graph: Graph, request: PathFindRequest) -> JSONResponse:
    """Find path(s) between two entities."""
    if request.find_all:
        result = find_all_paths(
            graph, request.source, request.target, request.max_length or 5
        )
    else:
        result = find_shortest_path(graph, request.source, request.target)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return JSONResponse(result)


@app.post("/api/analysis/neighbors")
async def get_neighbors(graph: Graph, request: NeighborRequest) -> JSONResponse:
    """Get neighbors of an entity."""
    result = get_entity_neighbors(graph, request.entity, request.depth or 1)

    if "error" in result:
        raise HTTPException(status_code=400, detail=result["error"])

    return JSONResponse(result)


@app.post("/api/analysis/full")
async def full_analysis(
    graph: Graph,
    top_k: int = Query(20, description="Number of top entities to return"),
) -> JSONResponse:
    """Compute comprehensive analysis of a graph."""
    analysis = compute_full_analysis(graph, top_k)
    return JSONResponse(analysis)


@app.get("/api/graphs/{graph_id}/analysis")
async def analyze_saved_graph(
    graph_id: str,
    top_k: int = Query(20),
) -> JSONResponse:
    """Compute full analysis for a saved graph."""
    graph = storage.load(graph_id)
    if graph is None:
        raise HTTPException(status_code=404, detail=f"Graph '{graph_id}' not found")

    analysis = compute_full_analysis(graph, top_k)
    return JSONResponse(analysis)


# =============================================================================
# Configuration Endpoint
# =============================================================================


@app.get("/api/config")
async def get_config() -> JSONResponse:
    """Get current configuration status."""
    azure_config = _get_azure_config()
    return JSONResponse({
        "azure_openai_configured": bool(azure_config),
        "azure_model": azure_config.get("model") if azure_config else None,
        "azure_deployment": azure_config.get("model", "").replace("azure/", "") if azure_config else None,
        "supported_document_formats": sorted(list(SUPPORTED_EXTENSIONS)),
        "storage_path": str(STORAGE_DIR),
    })


# Serve static files (CSS, JS, etc.) - must be mounted after all routes
app.mount("/", StaticFiles(directory=APP_DIR, html=True), name="static")
