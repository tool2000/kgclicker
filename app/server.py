from __future__ import annotations
from pathlib import Path
from typing import Optional

import logging
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from pydantic import ValidationError

from src.kg_gen.kg_gen import KGGen
from src.kg_gen.models import Graph
from src.kg_gen.utils.visualize_kg import _build_view_model

APP_DIR = Path(__file__).resolve().parent
TEMPLATE_PATH = (
    APP_DIR.parent / "src" / "kg_gen" / "utils" / "template.html"
).resolve()

if not TEMPLATE_PATH.exists():
    raise RuntimeError(f"Template not found at {TEMPLATE_PATH}")

logger = logging.getLogger("kg_gen_app")
if not logger.handlers:
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)


app = FastAPI(title="kg-gen explorer")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


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


def _clean_str(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    stripped = value.strip()
    return stripped or None


def _parse_bool(value: Optional[str]) -> bool:
    if value is None:
        return False
    return value.lower() in {"true", "1", "yes", "on"}


@app.post("/api/generate")
async def generate_graph(
    api_key: str = Form(..., description="OpenAI API key"),
    model: str = Form("openai/gpt-4o"),
    context: Optional[str] = Form(None),
    chunk_size: Optional[str] = Form(None),
    temperature: Optional[str] = Form(None),
    cluster: Optional[str] = Form(None),
    source_text: Optional[str] = Form(None),
    text_file: Optional[UploadFile] = File(None),
) -> JSONResponse:
    text_fragments: list[str] = []

    cleaned_text = _clean_str(source_text)
    if cleaned_text:
        text_fragments.append(cleaned_text)

    if text_file is not None:
        try:
            contents = await text_file.read()
        except Exception as exc:  # pragma: no cover - defensive
            logger.exception("Failed to read uploaded text file")
            raise HTTPException(
                status_code=400, detail=f"Reading text file failed: {exc}"
            )
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
            status_code=400, detail="Provide inline text or upload a .txt file"
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

    kg = KGGen(model=model, api_key=api_key)

    logger.info(
        "Generating graph via KGGen: model=%s cluster=%s chunk_size=%s context_len=%s text_len=%s",
        model,
        _parse_bool(cluster),
        numeric_chunk,
        len((_clean_str(context) or "")),
        len(request_text),
    )
    try:
        graph = kg.generate(
            input_data=request_text,
            model=model,
            api_key=api_key,
            context=_clean_str(context) or "",
            chunk_size=numeric_chunk,
            cluster=_parse_bool(cluster),
            temperature=numeric_temperature,
        )
    except ValidationError as exc:
        logger.exception("KGGen returned validation error")
        raise HTTPException(status_code=400, detail=f"Invalid graph result: {exc}")
    except Exception as exc:
        logger.exception("KGGen generation failed")
        raise HTTPException(status_code=500, detail=f"KGGen failed: {exc}")

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
    return JSONResponse(
        {
            "view": view,
            "graph": graph.model_dump(mode="json"),
        }
    )
