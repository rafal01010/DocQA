import logging
from pathlib import Path
import shutil
import uuid
from time import perf_counter

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from starlette.concurrency import run_in_threadpool

from backend.pipeline import OCRMarkdownPipeline

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
LOGGER = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parents[1]
UPLOAD_DIR = BASE_DIR / "data" / "uploads"
OCR_OUTPUT_DIR = BASE_DIR / "data" / "ocr_outputs"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
OCR_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="DocQA API")
ocr_pipeline = OCRMarkdownPipeline(output_dir=OCR_OUTPUT_DIR)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/uploads", StaticFiles(directory=str(UPLOAD_DIR)), name="uploads")
app.mount(
    "/ocr-outputs",
    StaticFiles(directory=str(OCR_OUTPUT_DIR)),
    name="ocr_outputs",
)


@app.get("/api/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/api/upload")
async def upload(file: UploadFile = File(...)) -> dict[str, str]:
    started = perf_counter()
    file_id = uuid.uuid4().hex
    original_name = file.filename or "document"
    suffix = Path(original_name).suffix
    stored_name = f"{file_id}{suffix}"
    destination = UPLOAD_DIR / stored_name
    LOGGER.info(
        "Upload received: file_id=%s original_name=%s suffix=%s destination=%s",
        file_id,
        original_name,
        suffix,
        destination,
    )

    try:
        with destination.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        LOGGER.info("Upload saved successfully: %s", destination)
    finally:
        await file.close()

    try:
        LOGGER.info("Starting OCR processing for file_id=%s", file_id)
        ocr_result = await run_in_threadpool(ocr_pipeline.process, file_id, destination)
        LOGGER.info(
            "OCR processing finished for file_id=%s markdown=%s text=%s",
            file_id,
            ocr_result.markdown_path,
            ocr_result.text_path,
        )
    except ValueError as exc:
        LOGGER.exception("Validation error during OCR for file_id=%s", file_id)
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        LOGGER.exception("Unexpected OCR failure for file_id=%s", file_id)
        raise HTTPException(status_code=500, detail=f"OCR failed: {exc}") from exc

    LOGGER.info("Upload request complete for file_id=%s elapsed=%.2fs", file_id, perf_counter() - started)
    return {
        "id": file_id,
        "filename": original_name,
        "url": f"/uploads/{stored_name}",
        "ocr_markdown_url": f"/ocr-outputs/{ocr_result.markdown_path.name}",
        "ocr_text_url": f"/ocr-outputs/{ocr_result.text_path.name}",
        "ocr_text_path": str(ocr_result.text_path),
    }
