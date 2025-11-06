from __future__ import annotations


def build_olmocr_markdown_prompt() -> str:
    """Build the prompt used for olmOCR-style markdown transcription."""
    try:
        from olmocr.prompts import build_no_anchoring_v4_yaml_prompt
    except ImportError:
        return (
            "You are a meticulous OCR assistant. Transcribe the provided document page into "
            "well-formatted Markdown that preserves headings, lists, and tables when possible. "
            "Do not include commentary outside the transcription."
        )
    return build_no_anchoring_v4_yaml_prompt()
