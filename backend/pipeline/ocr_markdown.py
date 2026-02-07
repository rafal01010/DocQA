from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from time import perf_counter

from backend.models import GlmOCRModel

LOGGER = logging.getLogger(__name__)


@dataclass(slots=True)
class OCRPipelineResult:
    markdown_path: Path
    text_path: Path


class OCRMarkdownPipeline:
    def __init__(
        self,
        output_dir: Path,
        model: GlmOCRModel | None = None,
    ) -> None:
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.model = model or GlmOCRModel()

    def process(self, document_id: str, source_path: Path) -> OCRPipelineResult:
        LOGGER.info("OCR pipeline start: document_id=%s source=%s", document_id, source_path)
        started = perf_counter()
        markdown = self.model.ocr_document(source_path)

        markdown_path = self.output_dir / f"{document_id}.md"
        markdown_path.write_text(markdown, encoding="utf-8")

        plain_text = self._markdown_to_text(markdown)
        text_path = self.output_dir / f"{document_id}.txt"
        text_path.write_text(plain_text, encoding="utf-8")

        LOGGER.info(
            "OCR pipeline complete: document_id=%s markdown=%s text=%s elapsed=%.2fs markdown_chars=%d text_chars=%d",
            document_id,
            markdown_path,
            text_path,
            perf_counter() - started,
            len(markdown),
            len(plain_text),
        )
        return OCRPipelineResult(markdown_path=markdown_path, text_path=text_path)

    @staticmethod
    def _markdown_to_text(markdown: str) -> str:
        lines = []
        for line in markdown.splitlines():
            if line.startswith("## Page "):
                continue
            lines.append(line)
        return "\n".join(lines).strip()
