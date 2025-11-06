from __future__ import annotations

import io
import shutil
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image


def discover_default_input(sample_dir: Path) -> Optional[Path]:
    """Return the first PDF file under sample_dir, if available."""
    if not sample_dir.exists():
        return None
    candidates = sorted(path for path in sample_dir.iterdir() if path.suffix.lower() == ".pdf")
    return candidates[0] if candidates else None


def prompt_user_for_output(target: Path) -> Tuple[Path, bool]:
    """Create or reuse an output directory, optionally skipping existing artifacts."""
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
        return target, False

    while True:
        response = input(
            f"Output directory '{target}' already exists. "
            "Skip existing [S] / overwrite [O]? (default: S): "
        ).strip().lower()

        if response in {"", "s", "skip", "skip existing"}:
            print("Existing output will be preserved; only missing pages will be generated.")
            return target, True

        if response in {"o", "overwrite"}:
            shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            return target, False

        print("Please respond with 's' to skip existing files or 'o' to overwrite.")


def import_mlx_stack():
    """Import MLX helpers and raise a friendly error when dependencies are missing."""
    try:
        import mlx.core as mx  # noqa: F401
        from mlx_vlm import generate as mlx_generate
        from mlx_vlm import load as mlx_load
        from mlx_vlm.prompt_utils import apply_chat_template as mlx_apply_chat_template
        from mlx_vlm.utils import load_config as mlx_load_config
    except ImportError as exc:  # pragma: no cover - imported lazily
        raise RuntimeError(
            "MLX vision-language dependencies are missing. Install them with `pip install mlx mlx_vlm`."
        ) from exc
    return mlx_load, mlx_generate, mlx_apply_chat_template, mlx_load_config


def resolve_model_path(
    model_root: Path, model_name: str, hf_fallback: Optional[str]
) -> Tuple[str, bool, Path]:
    """Return the loadable path (local or remote) for a model."""
    rooted = model_root.expanduser().resolve()
    local_candidate = rooted / model_name
    if local_candidate.exists():
        return str(local_candidate), True, local_candidate

    fallback = (hf_fallback or model_name).strip()
    if "/" not in fallback and not Path(fallback).expanduser().exists():
        fallback = f"mlx-community/{fallback}"
    return fallback, False, local_candidate


def downscale_for_vlm(
    img: Image.Image, max_side: int = 1280
) -> Tuple[Image.Image, bool, Tuple[int, int]]:
    """Resize images that are too large for the VLM while preserving aspect ratio."""
    original_size = (img.width, img.height)
    if img.mode != "RGB":
        img = img.convert("RGB")
    max_dim = max(original_size)
    if max_dim <= max_side:
        return img, False, original_size
    scale = max_side / float(max_dim)
    new_width = max(1, int(original_size[0] * scale))
    new_height = max(1, int(original_size[1] * scale))
    resized = img.resize((new_width, new_height), Image.BILINEAR)
    return resized, True, original_size


def render_pdf_to_images(pdf_path: Path, dpi: int, page_limit: Optional[int] = None) -> List[Image.Image]:
    """Render a PDF to a list of PIL.Image objects."""
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise RuntimeError("PyMuPDF is required for PDF rendering. Install it with `pip install pymupdf`.") from exc

    images: List[Image.Image] = []
    zoom = dpi / 72.0
    matrix = fitz.Matrix(zoom, zoom)

    with fitz.open(pdf_path) as pdf_doc:
        for page_index, page in enumerate(pdf_doc, start=1):
            if page_limit is not None and page_index > page_limit:
                break
            pix = page.get_pixmap(matrix=matrix, alpha=False)
            with io.BytesIO(pix.tobytes("png")) as buffer:
                image = Image.open(buffer)
                image.load()
                images.append(image.convert("RGB"))
    return images


def extract_text(result: Any) -> str:
    """Normalize the text output structure returned by mlx_vlm."""
    text_attr = getattr(result, "text", None)
    if isinstance(text_attr, str):
        return text_attr

    generated_attr = getattr(result, "generated_text", None)
    if isinstance(generated_attr, str):
        return generated_attr

    outputs_attr = getattr(result, "outputs", None)
    if isinstance(outputs_attr, (list, tuple)) and outputs_attr:
        first = outputs_attr[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            for key in ("generated_text", "text", "output"):
                value = first.get(key)
                if isinstance(value, str):
                    return value
        nested_text = getattr(first, "text", None)
        if isinstance(nested_text, str):
            return nested_text

    choices_attr = getattr(result, "choices", None)
    if isinstance(choices_attr, (list, tuple)) and choices_attr:
        first = choices_attr[0]
        if isinstance(first, dict):
            for key in ("text", "message", "generated_text"):
                value = first.get(key)
                if isinstance(value, str):
                    return value
                if isinstance(value, dict):
                    content_value = value.get("content")
                    if isinstance(content_value, str):
                        return content_value
        choice_text = getattr(first, "text", None)
        if isinstance(choice_text, str):
            return choice_text

    if hasattr(result, "to_dict"):
        try:
            result_dict = result.to_dict()
        except Exception:  # pragma: no cover - defensive
            result_dict = None
        if isinstance(result_dict, dict):
            for key in ("generated_text", "text", "output"):
                value = result_dict.get(key)
                if isinstance(value, str):
                    return value

    if isinstance(result, str):
        return result
    if isinstance(result, (list, tuple)):
        if not result:
            return ""
        first = result[0]
        if isinstance(first, str):
            return first
        if isinstance(first, dict):
            for key in ("generated_text", "text", "output"):
                value = first.get(key)
                if isinstance(value, str):
                    return value
        return str(first)
    if isinstance(result, dict):
        for key in ("generated_text", "text", "output"):
            value = result.get(key)
            if isinstance(value, str):
                return value
    return str(result)


def clean_generated_markdown(text: str) -> str:
    """Remove leading metadata artifacts from the generated Markdown."""
    if not text:
        return ""

    normalized = text.replace("\\n", "\n")
    lines = normalized.splitlines()
    if len(lines) >= 8:
        lines = lines[8:]
    else:
        lines = []

    cleaned = "\n".join(lines).strip()
    return cleaned


def generate_markdown_page(
    model: Any,
    processor: Any,
    config: Any,
    apply_template_fn: Callable[..., str],
    generate_fn: Callable[..., Any],
    image: Image.Image,
    prompt_text: str,
    max_tokens: Optional[int],
    temperature: Optional[float],
) -> str:
    """Produce a Markdown transcription for a single page image."""
    formatted_prompt = apply_template_fn(processor, config, prompt_text, num_images=1)
    result = generate_fn(
        model,
        processor,
        formatted_prompt,
        [image],
        verbose=False,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    print(result)
    return clean_generated_markdown(extract_text(result))
