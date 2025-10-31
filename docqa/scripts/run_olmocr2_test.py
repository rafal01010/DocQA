#!/usr/bin/env python3
"""
Quick test harness for running olmOCR-2 (MLX weights) against a sample PDF document.

- Prefers the locally cached `/Users/dave/models/olmOCR-2-7B-1025-4bit` folder,
  but falls back to the specified Hugging Face repo when the local path is missing.
- Uses `mlx_vlm` utilities so the model runs with Apple MLX acceleration (MPS/ANE).
- Accepts a PDF input and writes one Markdown file per page.
"""
from __future__ import annotations

import argparse
import io
import shutil
import sys
from pathlib import Path
from typing import Any, Callable, List, Optional, Tuple

from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLE_DIR = SCRIPT_DIR.parent / "sample_documents"
DEFAULT_MODEL_ROOT = Path("/Users/dave/AI/models")
DEFAULT_MODEL_NAME = "olmOCR-2-7B-1025-4bit"
DEFAULT_HF_FALLBACK = "mlx-community/olmOCR-2-7B-1025-4bit"


def _discover_default_input(sample_dir: Path = DEFAULT_SAMPLE_DIR) -> Optional[Path]:
    if not sample_dir.exists():
        return None
    candidates = sorted(path for path in sample_dir.iterdir() if path.suffix.lower() == ".pdf")
    return candidates[0] if candidates else None


def _prompt_user_for_output(target: Path) -> Path:
    if not target.exists():
        target.mkdir(parents=True, exist_ok=True)
        return target

    while True:
        response = input(
            f"Output directory '{target}' already exists. "
            "Overwrite [O] / keep both [K]? (default: O): "
        ).strip().lower()

        if response in {"", "o", "overwrite"}:
            shutil.rmtree(target)
            target.mkdir(parents=True, exist_ok=True)
            return target

        if response in {"k", "keep", "keep both"}:
            counter = 1
            while True:
                candidate = target.parent / f"{target.name}_{counter}"
                if not candidate.exists():
                    candidate.mkdir(parents=True, exist_ok=True)
                    print(f"Writing output to '{candidate}'.")
                    return candidate
                counter += 1

        print("Please respond with 'o' to overwrite or 'k' to keep both.")


def _import_mlx_stack():
    try:
        import mlx.core as mx  # noqa: F401
        from mlx_vlm import generate as mlx_generate
        from mlx_vlm import load as mlx_load
        from mlx_vlm.prompt_utils import apply_chat_template as mlx_apply_chat_template
        from mlx_vlm.utils import load_config as mlx_load_config
    except ImportError as exc:
        raise RuntimeError(
            "MLX vision-language dependencies are missing. "
            "Install them with `pip install mlx mlx_vlm`."
        ) from exc
    return mlx_load, mlx_generate, mlx_apply_chat_template, mlx_load_config


def _resolve_model_path(
    model_root: Path, model_name: str, hf_fallback: Optional[str]
) -> Tuple[str, bool, Path]:
    rooted = model_root.expanduser().resolve()
    local_candidate = rooted / model_name
    if local_candidate.exists():
        return str(local_candidate), True, local_candidate

    fallback = (hf_fallback or model_name).strip()
    if "/" not in fallback and not Path(fallback).expanduser().exists():
        fallback = f"mlx-community/{fallback}"
    return fallback, False, local_candidate


def _downscale_for_vlm(
    img: Image.Image, max_side: int = 1280
) -> Tuple[Image.Image, bool, Tuple[int, int]]:
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


def _render_pdf_to_images(pdf_path: Path, dpi: int, page_limit: Optional[int] = None) -> List[Image.Image]:
    try:
        import fitz  # PyMuPDF
    except ImportError as exc:
        raise RuntimeError(
            "PyMuPDF is required for PDF rendering. Install it with `pip install pymupdf`."
        ) from exc

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


def _build_prompt() -> str:
    try:
        from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
    except ImportError:
        return (
            "You are a meticulous OCR assistant. Transcribe the provided document page into "
            "well-formatted Markdown that preserves headings, lists, and tables when possible. "
            "Do not include commentary outside the transcription."
        )
    return build_no_anchoring_v4_yaml_prompt()


def _extract_text(result: Any) -> str:
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
        except Exception:
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


def _clean_generated_markdown(text: str) -> str:
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


def _generate_markdown_page(
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
    return _clean_generated_markdown(_extract_text(result))


def main() -> None:
    default_input = _discover_default_input()

    parser = argparse.ArgumentParser(
        description="Run local/remote MLX olmOCR-2 inference on a PDF and export Markdown per page."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=Path,
        default=default_input,
        required=default_input is None,
        help=(
            "Document to process (.pdf). "
            f"Defaults to '{default_input.name}' under ../sample_documents if omitted."
            if default_input
            else "Document to process (.pdf)."
        ),
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=Path,
        default=None,
        help="Folder (relative to docqa/scripts/results) where Markdown pages will be saved.",
    )
    parser.add_argument(
        "--model-root",
        type=Path,
        default=DEFAULT_MODEL_ROOT,
        help="Root directory containing locally downloaded model weights.",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default=DEFAULT_MODEL_NAME,
        help="Subdirectory name under the model root (default: olmOCR-2-7B-1025-4bit).",
    )
    parser.add_argument(
        "--hf-path",
        type=str,
        default=DEFAULT_HF_FALLBACK,
        help="Hugging Face repo to use when the local weights directory is unavailable.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=200,
        help="Rendering DPI for PDF pages (higher values improve fidelity at the cost of time).",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=8192,
        help="Generation cap per page (mapped to the closest supported mlx_vlm argument).",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Temperature for decoding when supported by the backend.",
    )
    parser.add_argument(
        "--max-side",
        type=int,
        default=1280,
        help="Largest allowed image dimension before downscaling for the VLM.",
    )
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Process every PDF page. Defaults to only the first page.",
    )

    args = parser.parse_args()

    input_path = args.input.expanduser().resolve()
    if not input_path.exists():
        print(f"Input file '{input_path}' not found.", file=sys.stderr)
        sys.exit(1)

    if input_path.suffix.lower() != ".pdf":
        print("Unsupported file type. Please provide a .pdf document.", file=sys.stderr)
        sys.exit(1)

    results_root = SCRIPT_DIR / "results"
    if args.output_dir:
        output_dir = (
            args.output_dir if args.output_dir.is_absolute() else results_root / args.output_dir
        )
    else:
        output_dir = results_root / f"{input_path.stem}_olmocr_pages"
    output_dir = output_dir.resolve()
    output_dir = _prompt_user_for_output(output_dir)

    page_limit = None if args.all_pages else 2

    try:
        images = _render_pdf_to_images(input_path, dpi=args.dpi, page_limit=page_limit)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if not images:
        print("No pages were rendered from the input document.", file=sys.stderr)
        sys.exit(1)

    try:
        mlx_load, mlx_generate, apply_template, load_config = _import_mlx_stack()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    model_path, used_local, local_candidate = _resolve_model_path(
        args.model_root, args.model_name, args.hf_path
    )
    if not used_local:
        print(
            f"Local weights not found at '{local_candidate}'. Falling back to '{model_path}'.",
            file=sys.stderr,
        )

    model = processor = None


    model, processor = mlx_load(model_path)

    if model is None or processor is None:
        print(f"Failed to load MLX model from '{model_path}' ", file=sys.stderr)
        sys.exit(1)

    try:
        config = load_config(model_path)
    except Exception as exc:  # pragma: no cover - runtime safety
        print(f"Failed to load MLX model config from '{model_path}': {exc}", file=sys.stderr)
        sys.exit(1)

    prompt_text = _build_prompt()

    for page_idx, image in enumerate(images, start=1):
        print(f"[{page_idx}/{len(images)}] Processing page...", flush=True)
        image, resized, original_size = _downscale_for_vlm(image, max_side=args.max_side)
        if resized:
            print(
                f"  -> Downscaled page image from {original_size[0]}x{original_size[1]} "
                f"to {image.width}x{image.height}"
            )
        try:
            markdown = _generate_markdown_page(
                model=model,
                processor=processor,
                config=config,
                apply_template_fn=apply_template,
                generate_fn=mlx_generate,
                image=image,
                prompt_text=prompt_text,
                max_tokens=args.max_tokens,
                temperature=args.temperature,
            )
        except Exception as exc:
            print(f"Failed to generate output for page {page_idx}: {exc}", file=sys.stderr)
            break

        output_file = output_dir / f"{input_path.stem}_page_{page_idx:03d}.md"
        if markdown and not markdown.endswith("\n"):
            markdown += "\n"
        output_file.write_text(markdown, encoding="utf-8")
        print(f"  -> Wrote {output_file.name}")


if __name__ == "__main__":
    main()
