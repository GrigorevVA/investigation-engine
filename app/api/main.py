"""HTTP API. Run: uvicorn app.api.main:app --port 7860  (UI at /, API docs at /docs)."""
import logging
from functools import lru_cache
from pathlib import Path

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, Field

from app.config import get_settings
from app.engine import Engine, build_engine
from app.ingest.loaders import UnsupportedFormatError, make_doc_id
from app.rag.llm import LLMError, LLMUnavailable
from app.search.vector_store import VectorStoreError

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
logging.getLogger("httpx").setLevel(logging.WARNING)  # one INFO line per Qdrant request is noise
MAX_UPLOAD_BYTES = 20 * 1024 * 1024
UI_PAGE = Path(__file__).parent.parent / "ui" / "index.html"

app = FastAPI(title="AI Investigation Engine", version="1.0")


@lru_cache
def get_engine() -> Engine:
    return build_engine(get_settings())


class SearchRequest(BaseModel):
    query: str = Field(min_length=1, max_length=1000)
    k: int = Field(default=8, ge=1, le=50)
    mode: str = Field(default="hybrid", pattern="^(vector|bm25|hybrid)$")


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=2000)
    mode: str | None = Field(default=None, pattern="^(vector|bm25|hybrid)$")


@app.exception_handler(LLMUnavailable)
def _llm_down(_, err: LLMUnavailable) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": f"The language model is unavailable right now. {err}"})


@app.exception_handler(LLMError)
def _llm_error(_, err: LLMError) -> JSONResponse:
    return JSONResponse(status_code=502, content={"detail": f"The language model rejected the request. {err}"})


@app.exception_handler(VectorStoreError)
def _store_down(_, err: VectorStoreError) -> JSONResponse:
    return JSONResponse(status_code=503, content={"detail": str(err)})


@app.get("/api/health")
def health(engine: Engine = Depends(get_engine)) -> dict:
    return {"status": "ok", **engine.stats()}


@app.get("/api/documents")
def list_documents(engine: Engine = Depends(get_engine)) -> list[dict]:
    return engine.list_documents()


async def _read_upload(file: UploadFile) -> bytes:
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        raise HTTPException(413, "File is larger than 20 MB")
    return data


@app.post("/api/documents", status_code=201)
async def upload_document(file: UploadFile = File(...), engine: Engine = Depends(get_engine)) -> dict:
    try:
        result = engine.add_file(await _read_upload(file), file.filename or "upload.txt")
    except UnsupportedFormatError as err:
        raise HTTPException(415, str(err)) from err
    except ValueError as err:
        raise HTTPException(422, str(err)) from err
    return {"doc_id": result.doc_id, "status": result.status, "chunks": result.chunks,
            "duplicate_of": result.duplicate_of}


@app.put("/api/documents/{doc_id:path}")
async def update_document(doc_id: str, file: UploadFile = File(...), engine: Engine = Depends(get_engine)) -> dict:
    """Update = upload a new version under the same file name."""
    if make_doc_id(file.filename or "") != doc_id:
        raise HTTPException(400, f"File name maps to {make_doc_id(file.filename or '')!r}, not {doc_id!r}")
    return await upload_document(file, engine)


@app.delete("/api/documents/{doc_id:path}")
def delete_document(doc_id: str, engine: Engine = Depends(get_engine)) -> dict:
    if not engine.delete(doc_id):
        raise HTTPException(404, f"Document {doc_id!r} not found")
    return {"doc_id": doc_id, "status": "deleted"}


@app.post("/api/search")
def search(req: SearchRequest, engine: Engine = Depends(get_engine)) -> list[dict]:
    return [{"chunk_id": h.chunk.chunk_id, "doc_id": h.chunk.doc_id, "title": h.chunk.title,
             "score": round(h.score, 4), "retriever": h.retriever, "text": h.chunk.text}
            for h in engine.search(req.query, req.k, req.mode)]


@app.post("/api/ask")
def ask(req: AskRequest, engine: Engine = Depends(get_engine)) -> dict:
    return engine.investigate(req.question, req.mode).to_dict()


@app.get("/", include_in_schema=False)
def ui() -> FileResponse:
    """Plain HTML + JS page over this API. Replaced Gradio, which alone took ~120 MB of a 512 MB host."""
    return FileResponse(UI_PAGE, media_type="text/html")
