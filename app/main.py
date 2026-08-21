"""
FastAPI application — main entry point.
Serves API endpoints and the static frontend.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from app.config import get_config, get_secrets, PROJECT_ROOT, load_config, load_secrets
from app.models import (
    ChunkStrategy,
    HealthResponse,
    PipelineResponse,
    TextQueryRequest,
)
from app.pipeline import RAGPipeline, get_pipeline
from app.stages.embedder import get_embedding_model
from app.stages.retriever import get_retriever

# ── Logging Setup ─────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("rag.api")

# ── FastAPI App ───────────────────────────────────────────────────────────

app = FastAPI(
    title="Lumora — Voice-Enabled Multilingual RAG",
    description="Multilingual RAG pipeline with Sarvam STT, FAISS retrieval, and Groq LLM generation",
    version="1.0.0",
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Startup Event ─────────────────────────────────────────────────────────

@app.on_event("startup")
async def startup_event():
    """Pre-warm models and load indices at startup."""
    logger.info("=" * 60)
    logger.info("Starting Voice-Enabled RAG System")
    logger.info("=" * 60)

    # Load config
    load_config()
    load_secrets()

    config = get_config()
    secrets = get_secrets()

    # Pre-warm embedding model
    logger.info("Loading embedding model...")
    embedder = get_embedding_model()
    embedder.load()

    # Load FAISS index
    logger.info("Loading FAISS index...")
    retriever = get_retriever()
    retriever.load()

    # Initialize pipeline
    logger.info("Initializing pipeline...")
    _ = get_pipeline()

    logger.info("=" * 60)
    logger.info(f"System ready | Index size: {retriever.index_size} chunks")
    logger.info(f"Available strategies: {retriever.available_strategies}")
    logger.info(f"Sarvam configured: {bool(secrets.sarvam_api_key)}")
    logger.info(f"Groq configured: {bool(secrets.groq_api_key)}")
    logger.info("=" * 60)


# ── API Endpoints ─────────────────────────────────────────────────────────

@app.get("/api/health", response_model=HealthResponse)
async def health_check():
    """Health check with model and index status."""
    embedder = get_embedding_model()
    retriever = get_retriever()
    secrets = get_secrets()

    return HealthResponse(
        status="healthy" if embedder.is_loaded and retriever.is_loaded else "degraded",
        embedding_model_loaded=embedder.is_loaded,
        index_loaded=retriever.is_loaded,
        index_size=retriever.index_size,
        available_strategies=retriever.available_strategies,
        sarvam_configured=bool(secrets.sarvam_api_key),
        groq_configured=bool(secrets.groq_api_key),
    )


@app.post("/api/v1/query", response_model=PipelineResponse)
@app.post("/api/query/text", response_model=PipelineResponse)
async def query_text(request: TextQueryRequest):
    """
    Text-based query endpoint. Skips STT, runs RAG pipeline.
    Pipeline latency target: <200ms (embed + retrieve + generate + guardrail).
    """
    pipeline = get_pipeline()

    try:
        response = await pipeline.process_text(
            query=request.query,
            strategy=request.strategy,
            top_k=request.top_k,
        )
        return response
    except Exception as e:
        logger.error(f"Text query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/v1/query/voice", response_model=PipelineResponse)
@app.post("/api/query/voice", response_model=PipelineResponse)
async def query_voice(
    audio: UploadFile = File(...),
    strategy: str = Form(default="fixed"),
    top_k: int = Form(default=5),
):
    """
    Voice-based query endpoint. Runs full pipeline: STT → RAG → response.
    STT latency is measured and reported separately from the pipeline target.
    """
    pipeline = get_pipeline()

    try:
        # Validate strategy
        try:
            chunk_strategy = ChunkStrategy(strategy)
        except ValueError:
            chunk_strategy = ChunkStrategy.FIXED

        # Read audio bytes
        audio_bytes = await audio.read()
        if not audio_bytes:
            raise HTTPException(status_code=400, detail="Empty audio file")

        response = await pipeline.process_voice(
            audio_bytes=audio_bytes,
            filename=audio.filename or "audio.wav",
            strategy=chunk_strategy,
            top_k=top_k,
        )
        return response
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Voice query failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/strategies")
async def get_strategies():
    """Get available chunking strategies and their stats."""
    retriever = get_retriever()
    strategies = retriever.available_strategies

    stats = {}
    for s in strategies:
        indices = retriever._strategy_indices.get(s, [])
        stats[s] = {
            "name": s,
            "chunk_count": len(indices),
            "description": {
                "fixed": "Fixed-size chunks with 20% overlap (256 tokens)",
                "semantic": "Chunks split at topic-shift boundaries using embedding similarity",
                "sentence_window": "Small anchor chunks with expanded context window",
                "metadata_aware": "Full passages with preserved MSMARCO metadata",
            }.get(s, "Unknown strategy"),
        }

    return {"strategies": stats, "default": get_config().chunking.default_strategy}


# ── Static Files (Frontend) ──────────────────────────────────────────────

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/")
async def serve_frontend():
    """Serve the frontend UI."""
    index_path = static_dir / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return JSONResponse(
        {"message": "Voice-Enabled RAG API", "docs": "/docs"},
        status_code=200,
    )


@app.get("/{file_name}")
async def serve_root_files(file_name: str):
    """Serve root static files like styles.css, main.js, etc."""
    file_path = static_dir / file_name
    if file_path.exists() and file_path.is_file():
        return FileResponse(str(file_path))
    raise HTTPException(status_code=404, detail="File not found")


@app.get("/assets/{asset_path:path}")
async def serve_assets(asset_path: str):
    """Serve assets directory."""
    asset_file = static_dir / "assets" / asset_path
    if asset_file.exists() and asset_file.is_file():
        return FileResponse(str(asset_file))
    # Fallback to root assets folder if exists
    root_asset = PROJECT_ROOT / "assets" / asset_path
    if root_asset.exists() and root_asset.is_file():
        return FileResponse(str(root_asset))
    raise HTTPException(status_code=404, detail="Asset not found")


@app.get("/fonts/{font_path:path}")
async def serve_fonts(font_path: str):
    """Serve fonts directory."""
    font_file = static_dir / "fonts" / font_path
    if font_file.exists() and font_file.is_file():
        return FileResponse(str(font_file))
    root_font = PROJECT_ROOT / "fonts" / font_path
    if root_font.exists() and root_font.is_file():
        return FileResponse(str(root_font))
    raise HTTPException(status_code=404, detail="Font not found")


# ── Run with uvicorn ──────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    config = get_config()
    uvicorn.run(
        "app.main:app",
        host=config.server.host,
        port=config.server.port,
        log_level=config.server.log_level,
        reload=False,
    )
