from __future__ import annotations

import json
import logging
from pathlib import Path
from tempfile import NamedTemporaryFile, TemporaryDirectory
from time import perf_counter

import fitz
import torch
from PIL import Image
from transformers import AutoModelForImageTextToText, AutoProcessor

LOGGER = logging.getLogger(__name__)


class GlmOCRModel:
    """Wrapper around GLM OCR inference for images and PDFs."""

    DEFAULT_MODEL_PATH = Path("/Users/dave/AI/models/GLM-OCR")
    IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}

    def __init__(
        self,
        model_path: Path | str | None = None,
        prompt: str = "Text Recognition:",
        max_new_tokens: int = 8192,
        max_image_side: int = 1024,
        max_pdf_pages: int | None = 5,
    ) -> None:
        self.model_path = Path(model_path) if model_path else self.DEFAULT_MODEL_PATH
        self.prompt = prompt
        self.max_new_tokens = max_new_tokens
        self.max_image_side = max_image_side
        self.max_pdf_pages = max_pdf_pages
        self._processor = None
        self._model = None

    @staticmethod
    def _select_device_and_dtype() -> tuple[torch.device, torch.dtype]:
        if torch.backends.mps.is_available():
            return torch.device("mps"), torch.float16

        if torch.cuda.is_available():
            return torch.device("cuda"), torch.float16

        return torch.device("cpu"), torch.float32

    def _load(self) -> None:
        if self._processor is not None and self._model is not None:
            return

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {self.model_path}")

        device, dtype = self._select_device_and_dtype()
        LOGGER.info(
            "Loading GLM OCR model from '%s' on device=%s dtype=%s",
            self.model_path,
            device,
            dtype,
        )
        if torch.backends.mps.is_built():
            LOGGER.info("PyTorch Metal backend is built: true")
        else:
            LOGGER.info("PyTorch Metal backend is built: false")

        try:
            self._processor = AutoProcessor.from_pretrained(
                str(self.model_path),
                trust_remote_code=True,
            )
            LOGGER.info("GLM OCR processor loaded successfully")
        except Exception as exc:
            model_snapshot = self._model_dir_snapshot()
            raise RuntimeError(
                "Failed to load GLM OCR processor. "
                f"Model directory snapshot: {model_snapshot}"
            ) from exc

        try:
            self._model = AutoModelForImageTextToText.from_pretrained(
                pretrained_model_name_or_path=str(self.model_path),
                torch_dtype=dtype,
                trust_remote_code=True,
            )
            LOGGER.info("GLM OCR model weights loaded successfully")
        except Exception as exc:
            model_snapshot = self._model_dir_snapshot()
            raise RuntimeError(
                "Failed to load GLM OCR model weights. "
                f"Model directory snapshot: {model_snapshot}"
            ) from exc

        self._model.to(device)
        self._model.eval()
        LOGGER.info("GLM OCR model is ready")

    def _run(self, image_path: Path, prompt: str | None = None) -> str:
        self._load()

        assert self._processor is not None
        assert self._model is not None

        effective_prompt = prompt or self.prompt
        prepared_image_path, cleanup_path = self._prepare_image_for_inference(image_path)
        LOGGER.info(
            "Running OCR on image '%s' (prepared='%s') with prompt '%s'",
            image_path,
            prepared_image_path,
            effective_prompt,
        )
        message = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "url": str(prepared_image_path)},
                    {"type": "text", "text": effective_prompt},
                ],
            }
        ]

        try:
            started = perf_counter()
            inputs = self._processor.apply_chat_template(
                message,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            ).to(self._model.device)
            inputs.pop("token_type_ids", None)

            with torch.no_grad():
                generated_ids = self._model.generate(
                    **inputs,
                    max_new_tokens=self.max_new_tokens,
                )

            input_length = inputs["input_ids"].shape[1]
            text = self._processor.decode(
                generated_ids[0][input_length:],
                skip_special_tokens=True,
            )
            LOGGER.info(
                "OCR image complete for '%s' in %.2fs (output_chars=%d)",
                image_path.name,
                perf_counter() - started,
                len(text),
            )
            return text.strip()
        finally:
            if cleanup_path is not None and cleanup_path.exists():
                cleanup_path.unlink(missing_ok=True)

    def ocr_image(self, image_path: Path | str, prompt: str | None = None) -> str:
        image = Path(image_path)
        if not image.exists():
            raise FileNotFoundError(f"Image not found: {image}")
        return self._run(image, prompt=prompt)

    def ocr_pdf(self, pdf_path: Path | str, prompt: str | None = None, dpi: int = 200) -> str:
        pdf = Path(pdf_path)
        if not pdf.exists():
            raise FileNotFoundError(f"PDF not found: {pdf}")

        LOGGER.info("Starting PDF OCR for '%s' at dpi=%d", pdf, dpi)
        total_started = perf_counter()
        zoom = dpi / 72
        matrix = fitz.Matrix(zoom, zoom)
        sections: list[str] = []

        with fitz.open(pdf) as document, TemporaryDirectory(prefix="docqa_ocr_") as temp_dir:
            LOGGER.info("PDF page count: %d", document.page_count)
            pages_to_process = document.page_count
            if self.max_pdf_pages is not None:
                pages_to_process = min(document.page_count, self.max_pdf_pages)
            LOGGER.info(
                "PDF processing page limit applied: processing %d/%d pages",
                pages_to_process,
                document.page_count,
            )
            temp_path = Path(temp_dir)
            for index, page in enumerate(document, start=1):
                if index > pages_to_process:
                    break
                page_started = perf_counter()
                image_path = temp_path / f"page_{index:04d}.png"
                pixmap = page.get_pixmap(matrix=matrix, alpha=False)
                pixmap.save(image_path)

                page_text = self._run(image_path, prompt=prompt)
                sections.append(f"## Page {index}\n\n{page_text}")
                LOGGER.info(
                    "Finished page %d/%d in %.2fs",
                    index,
                    document.page_count,
                    perf_counter() - page_started,
                )

        LOGGER.info(
            "Completed PDF OCR for '%s' in %.2fs",
            pdf.name,
            perf_counter() - total_started,
        )
        return "\n\n".join(sections).strip()

    def ocr_document(self, document_path: Path | str, prompt: str | None = None) -> str:
        path = Path(document_path)
        suffix = path.suffix.lower()

        if suffix == ".pdf":
            return self.ocr_pdf(path, prompt=prompt)

        if suffix in self.IMAGE_EXTENSIONS:
            text = self.ocr_image(path, prompt=prompt)
            return f"## Page 1\n\n{text}".strip()

        raise ValueError(
            f"Unsupported document type '{suffix}'. Supported: PDF and image files."
        )

    def _model_dir_snapshot(self) -> str:
        files = sorted(path.name for path in self.model_path.iterdir()) if self.model_path.exists() else []
        preview = ", ".join(files[:20])
        if len(files) > 20:
            preview = f"{preview}, ..."

        config_hint = ""
        config_path = self.model_path / "config.json"
        if config_path.exists():
            try:
                config = json.loads(config_path.read_text(encoding="utf-8"))
                model_type = config.get("model_type", "<missing>")
                auto_map = config.get("auto_map", {})
                config_hint = f" model_type={model_type} auto_map_keys={list(auto_map.keys())}"
            except Exception:
                config_hint = " config_unreadable=true"

        return f"files=[{preview}]{config_hint}"

    def _prepare_image_for_inference(self, image_path: Path) -> tuple[Path, Path | None]:
        with Image.open(image_path) as image:
            width, height = image.size
            max_side = max(width, height)
            if max_side <= self.max_image_side:
                LOGGER.info(
                    "Image '%s' kept at %dx%d (max_side=%d <= limit=%d)",
                    image_path.name,
                    width,
                    height,
                    max_side,
                    self.max_image_side,
                )
                return image_path, None

            scale = self.max_image_side / max_side
            new_width = max(1, int(width * scale))
            new_height = max(1, int(height * scale))
            resized = image.convert("RGB").resize(
                (new_width, new_height),
                Image.Resampling.LANCZOS,
            )

            with NamedTemporaryFile(prefix="docqa_ocr_resized_", suffix=".jpg", delete=False) as tmp:
                resized_path = Path(tmp.name)
            resized.save(resized_path, format="JPEG", quality=90, optimize=True)

            LOGGER.info(
                "Image '%s' resized from %dx%d to %dx%d for OCR",
                image_path.name,
                width,
                height,
                new_width,
                new_height,
            )
            return resized_path, resized_path
