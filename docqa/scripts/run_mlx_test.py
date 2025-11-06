#!/usr/bin/env python3
"""
Generic MLX VLM test harness that renders a PDF and produces Markdown per page.

- Defaults to the locally cached `/Users/dave/models/olmOCR-2-7B-1025-4bit` weights,
  but will fall back to the specified Hugging Face repo when the local path is missing.
- Uses `mlx_vlm` utilities so the model runs with Apple MLX acceleration (MPS/ANE).
- Auto-selects model-specific helpers (e.g. prompts) based on the model identifier.
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Callable, Optional, Sequence, Tuple

from utils.general import (
    discover_default_input,
    downscale_for_vlm,
    generate_markdown_page,
    import_mlx_stack,
    prompt_user_for_output,
    render_pdf_to_images,
    resolve_model_path,
)
from utils.olmocr import build_olmocr_markdown_prompt

SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_SAMPLE_DIR = SCRIPT_DIR.parent / "sample_documents"
DEFAULT_MODEL_ROOT = Path("/Users/dave/AI/models")
DEFAULT_MODEL_NAME = "olmOCR-2-7B-1025-4bit"
DEFAULT_HF_FALLBACK = "mlx-community/olmOCR-2-7B-1025-4bit"
DEFAULT_PROMPT = (
    "You are a meticulous OCR assistant. Transcribe the provided document page into "
    "well-formatted Markdown that preserves headings, lists, and tables when possible. "
    "Do not include commentary outside the transcription."
)

PROMPT_REGISTRY: Sequence[Tuple[str, Callable[[], str]]] = (
    ("olmocr", build_olmocr_markdown_prompt),
)


def slugify(value: str) -> str:
    cleaned = re.sub(r"[^\w]+", "_", value or "")
    cleaned = cleaned.strip("_").lower()
    return cleaned or "model"


def resolve_prompt_text(*identifiers: str) -> str:
    for identifier in identifiers:
        if not identifier:
            continue
        lowered = identifier.lower()
        for keyword, builder in PROMPT_REGISTRY:
            if keyword in lowered:
                return builder()
    return DEFAULT_PROMPT


def main() -> None:
    default_input = discover_default_input(DEFAULT_SAMPLE_DIR)

    parser = argparse.ArgumentParser(
        description="Run local/remote MLX VLM inference on a PDF and export Markdown per page."
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
        "--page-limit",
        type=int,
        default=1,
        help="Maximum number of pages to process (ignored when --all-pages is set).",
    )
    parser.add_argument(
        "--all-pages",
        action="store_true",
        help="Process every PDF page. Overrides --page-limit.",
    )
    parser.add_argument(
        "--save-images",
        action="store_true",
        help="Also export each rendered PDF page as a PNG alongside the Markdown output.",
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
        model_slug = slugify(args.model_name or DEFAULT_MODEL_NAME)
        output_dir = results_root / f"{input_path.stem}_{model_slug}_pages"
    output_dir = output_dir.resolve()
    output_dir, skip_existing = prompt_user_for_output(output_dir)

    if not args.all_pages and args.page_limit < 1:
        print("--page-limit must be at least 1 when --all-pages is not provided.", file=sys.stderr)
        sys.exit(1)

    page_limit = None if args.all_pages else args.page_limit

    try:
        images = render_pdf_to_images(input_path, dpi=args.dpi, page_limit=page_limit)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    if not images:
        print("No pages were rendered from the input document.", file=sys.stderr)
        sys.exit(1)

    try:
        mlx_load, mlx_generate, apply_template, load_config = import_mlx_stack()
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        sys.exit(1)

    model_path, used_local, local_candidate = resolve_model_path(
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

    prompt_text = resolve_prompt_text(args.model_name, model_path)

    for page_idx, page_image in enumerate(images, start=1):
        markdown_file = output_dir / f"{input_path.stem}_page_{page_idx:03d}.md"
        image_file: Optional[Path] = (
            output_dir / f"{input_path.stem}_page_{page_idx:03d}.png" if args.save_images else None
        )

        markdown_exists = markdown_file.exists()
        image_exists = image_file.exists() if image_file else False

        skip_page_entirely = (
            skip_existing
            and markdown_exists
            and (not image_file or image_exists)
        )
        if skip_page_entirely:
            if image_file:
                print(
                    f"[{page_idx}/{len(images)}] Skipping existing output "
                    f"({markdown_file.name}, {image_file.name}).",
                    flush=True,
                )
            else:
                print(
                    f"[{page_idx}/{len(images)}] Skipping existing output ({markdown_file.name}).",
                    flush=True,
                )
            continue

        should_generate_markdown = not (skip_existing and markdown_exists)
        if should_generate_markdown:
            print(f"[{page_idx}/{len(images)}] Processing page...", flush=True)
        else:
            print(
                f"[{page_idx}/{len(images)}] Updating page artifacts (markdown already present)...",
                flush=True,
            )

        image_for_model = page_image
        resized = False
        original_size = (page_image.width, page_image.height)
        if should_generate_markdown:
            image_for_model, resized, original_size = downscale_for_vlm(
                page_image, max_side=args.max_side
            )
            if resized:
                print(
                    f"  -> Downscaled page image from {original_size[0]}x{original_size[1]} "
                    f"to {image_for_model.width}x{image_for_model.height}"
                )

        markdown: Optional[str] = None
        if should_generate_markdown:
            try:
                markdown = generate_markdown_page(
                    model=model,
                    processor=processor,
                    config=config,
                    apply_template_fn=apply_template,
                    generate_fn=mlx_generate,
                    image=image_for_model,
                    prompt_text=prompt_text,
                    max_tokens=args.max_tokens,
                    temperature=args.temperature,
                )
            except Exception as exc:
                print(f"Failed to generate output for page {page_idx}: {exc}", file=sys.stderr)
                break

        if should_generate_markdown:
            if markdown and not markdown.endswith("\n"):
                markdown += "\n"
            markdown_file.write_text(markdown, encoding="utf-8")
            print(f"  -> Wrote {markdown_file.name}")
        else:
            print(f"  -> Markdown already exists; reusing {markdown_file.name}")

        if image_file:
            should_save_image = not (skip_existing and image_exists)
            if should_save_image:
                page_image_to_save = (
                    page_image if page_image.mode == "RGB" else page_image.convert("RGB")
                )
                page_image_to_save.save(image_file, format="PNG")
                print(f"  -> Wrote {image_file.name}")
            else:
                print(f"  -> Image already exists; reusing {image_file.name}")


if __name__ == "__main__":
    main()
