"""
OCR Engine wrapper for PaddleOCR.
Isolates all OCR operations, handles errors gracefully, and prevents memory leaks.
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from PIL import Image

logger = logging.getLogger(__name__)

# Configure environment variables before paddle loads to prevent CPU oneDNN/PIR issues
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
os.environ["PADDLEX_DISABLE_AUTO_INIT"] = "True"
os.environ["PADDLEX_LOG_LEVEL"] = "CRITICAL"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


@dataclass
class OCRLine:
    text: str
    confidence: float
    box: Optional[Any] = None


@dataclass
class OCRResult:
    text: str
    lines: List[OCRLine] = field(default_factory=list)
    mean_confidence: float = 0.0
    success: bool = True
    error: Optional[str] = None

    def get_safe_preview(self, max_chars: int = 60) -> str:
        """
        Return a sanitized preview of OCR text that masks 10-12 digit numbers
        to avoid exposing sensitive Aadhaar / PAN / Account numbers in logs.
        """
        if not self.text:
            return "<No Text Extracted>"
        masked = re.sub(r"\b\d{4}\s?\d{4}\s?\d{4}\b", "XXXX-XXXX-XXXX", self.text)
        masked = re.sub(r"\b[A-Z]{5}\d{4}[A-Z]\b", "XXXXX0000X", masked)
        masked = re.sub(r"\b\d{9,18}\b", "XXXXXXXXXX", masked)
        preview = " ".join(masked.split())
        if len(preview) > max_chars:
            return preview[:max_chars] + "..."
        return preview


class OCREngine:
    """
    Isolated PaddleOCR wrapper with lazy loading, error recovery,
    and automatic compatibility handling across PaddlePaddle versions.
    """

    def __init__(
        self,
        language: str = "en",
        use_gpu: bool = False,
        cpu_threads: int = 4,
    ):
        self.language = language
        self.use_gpu = use_gpu
        self.cpu_threads = cpu_threads
        self._ocr = None

    def _get_ocr_instance(self):
        """Lazy initialization of PaddleOCR instance with safety patches."""
        if self._ocr is None:
            logger.info("Initializing PaddleOCR engine (lang=%s, gpu=%s)...", self.language, self.use_gpu)
            os.environ["OMP_NUM_THREADS"] = "1"
            os.environ["MKL_NUM_THREADS"] = "1"
            os.environ["OPENBLAS_NUM_THREADS"] = "1"

            # PaddleX oneDNN CPU patch:
            # Paddle 3.3.1 static runner has a known oneDNN instruction bug when run_mode='mkldnn'.
            # Patching runner run_mode to 'paddle' uses OpenMP BLAS CPU acceleration cleanly.
            try:
                import paddlex.inference.models.runners.paddle_static.runner as pdx_runner
                if hasattr(pdx_runner, "PaddleStaticRunner"):
                    orig_create = pdx_runner.PaddleStaticRunner._create

                    def patched_create(runner_self):
                        runner_self._config["run_mode"] = "paddle"
                        runner_self._config["enable_new_ir"] = False
                        return orig_create(runner_self)

                    pdx_runner.PaddleStaticRunner._create = patched_create
            except Exception as e:
                logger.debug("PaddleStaticRunner patch skipped or not applicable: %s", e)

            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(
                    lang=self.language,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=False,
                )
                logger.info("PaddleOCR engine initialized successfully.")
            except Exception as e:
                logger.error("Failed to initialize PaddleOCR: %s", e)
                raise
        return self._ocr

    def extract_text(self, image_path: Union[str, Path]) -> OCRResult:
        """
        Extract text from an image file safely. Never raises unhandled exceptions to caller.
        """
        img_path = Path(image_path)
        if not img_path.is_file():
            return OCRResult(
                text="",
                success=False,
                error=f"File not found: {img_path}",
            )

        try:
            with Image.open(img_path) as img:
                img.verify()
        except Exception as e:
            return OCRResult(
                text="",
                success=False,
                error=f"Corrupt or unreadable image file: {e}",
            )

        try:
            ocr = self._get_ocr_instance()
            # Try predict first (PaddleOCR v3.x), fallback to ocr()
            if hasattr(ocr, "predict"):
                raw_output = list(ocr.predict(str(img_path)))
            else:
                raw_output = ocr.ocr(str(img_path))
        except Exception as e:
            logger.error("PaddleOCR execution failed on '%s': %s", img_path.name, e)
            return OCRResult(
                text="",
                success=False,
                error=f"OCR execution error: {e}",
            )

        extracted_lines: List[OCRLine] = []
        full_text_parts: List[str] = []
        conf_scores: List[float] = []

        if raw_output:
            pages = raw_output if isinstance(raw_output, list) else [raw_output]
            for page in pages:
                if not page:
                    continue

                # Format 1: PaddleX Dictionary output (v3.x / v6 models)
                if isinstance(page, dict):
                    rec_texts = page.get("rec_texts", [])
                    rec_scores = page.get("rec_scores", [])
                    rec_polys = page.get("rec_polys", [])
                    for i, txt in enumerate(rec_texts):
                        txt_str = str(txt).strip()
                        if not txt_str:
                            continue
                        score = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                        poly = rec_polys[i] if i < len(rec_polys) else None
                        extracted_lines.append(OCRLine(text=txt_str, confidence=score, box=poly))
                        full_text_parts.append(txt_str)
                        conf_scores.append(score)

                # Format 2: Standard legacy list format [[box, (text, conf)], ...]
                elif isinstance(page, list):
                    for item in page:
                        if isinstance(item, (list, tuple)) and len(item) >= 2:
                            box = item[0] if isinstance(item[0], list) else None
                            content = item[1]
                            if isinstance(content, (list, tuple)) and len(content) >= 2:
                                txt, conf = str(content[0]).strip(), float(content[1])
                            else:
                                txt, conf = str(content).strip(), 1.0
                            if txt:
                                extracted_lines.append(OCRLine(text=txt, confidence=conf, box=box))
                                full_text_parts.append(txt)
                                conf_scores.append(conf)
                        elif isinstance(item, dict):
                            txt = str(item.get("text", "")).strip()
                            conf = float(item.get("confidence", item.get("score", 1.0)))
                            if txt:
                                extracted_lines.append(OCRLine(text=txt, confidence=conf))
                                full_text_parts.append(txt)
                                conf_scores.append(conf)

        combined_text = "\n".join(full_text_parts)
        mean_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

        return OCRResult(
            text=combined_text,
            lines=extracted_lines,
            mean_confidence=mean_conf,
            success=True,
            error=None,
        )
