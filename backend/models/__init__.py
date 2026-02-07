from importlib.util import module_from_spec, spec_from_file_location
from pathlib import Path

_MODULE_PATH = Path(__file__).with_name("glm-ocr.py")
_SPEC = spec_from_file_location("backend.models.glm_ocr_impl", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise RuntimeError(f"Unable to load OCR module from {_MODULE_PATH}")

_MODULE = module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)

GlmOCRModel = _MODULE.GlmOCRModel

__all__ = ["GlmOCRModel"]
