"""Utility helpers shared across MLX demo scripts."""

from .general import (
    clean_generated_markdown,
    discover_default_input,
    downscale_for_vlm,
    generate_markdown_page,
    import_mlx_stack,
    prompt_user_for_output,
    render_pdf_to_images,
    resolve_model_path,
)
from .olmocr import build_olmocr_markdown_prompt

__all__ = [
    "clean_generated_markdown",
    "discover_default_input",
    "downscale_for_vlm",
    "generate_markdown_page",
    "import_mlx_stack",
    "prompt_user_for_output",
    "render_pdf_to_images",
    "resolve_model_path",
    "build_olmocr_markdown_prompt",
]
