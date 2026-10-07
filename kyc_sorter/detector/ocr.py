"""
OCR Engine wrapper for PaddleOCR.
Reuses the exact OCR pipeline and model configuration from Finverge KYC backend:
- PP-OCRv5_mobile_det (Detection)
- en_PP-OCRv5_mobile_rec (Recognition)
- CPU-only execution with configurable thread limits
- Single-pass extraction with safe image preprocessing and parsing
"""

from __future__ import annotations

import logging
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import numpy as np
from PIL import Image

try:
    import cv2
except ImportError:
    cv2 = None

logger = logging.getLogger(__name__)

# Configure environment variables before paddle loads
os.environ["FLAGS_enable_pir_api"] = "0"
os.environ["PADDLE_DISABLE_ONEDNN"] = "1"
os.environ["PADDLEX_DISABLE_AUTO_INIT"] = "True"
os.environ["PADDLEX_LOG_LEVEL"] = "CRITICAL"
os.environ["PADDLE_PDX_DISABLE_MODEL_SOURCE_CHECK"] = "True"


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
    file_load_time_sec: float = 0.0
    preprocessing_time_sec: float = 0.0
    ocr_init_time_sec: float = 0.0
    ocr_inference_time_sec: float = 0.0
    ocr_extract_time_sec: float = 0.0
    crop_count: int = 0
    rec_model_name: str = "en_PP-OCRv5_mobile_rec"

    @property
    def init_time_sec(self) -> float:
        return self.ocr_init_time_sec

    @property
    def inference_time_sec(self) -> float:
        return self.ocr_inference_time_sec

    @property
    def postprocessing_time_sec(self) -> float:
        return self.ocr_extract_time_sec

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
            preview = preview[:max_chars] + "..."
        return preview.encode("ascii", errors="replace").decode("ascii")


class OCREngine:
    """
    PaddleOCR wrapper reusing the exact KYC backend OCR configuration:
    PP-OCRv5_mobile_det and en_PP-OCRv5_mobile_rec on CPU.
    """
    _shared_instances: Dict[str, Any] = {}
    _shared_ocr: Optional[Any] = None

    def __init__(
        self,
        language: str = "en",
        use_gpu: bool = False,
        cpu_threads: int = 3,
        det_model_name: str = "PP-OCRv5_mobile_det",
        rec_model_name: str = "en_PP-OCRv5_mobile_rec",
        enable_mkldnn: bool = False,
        use_angle_cls: bool = False,
    ):
        self.language = language
        self.use_gpu = use_gpu
        self.cpu_threads = max(1, cpu_threads)
        self.det_model_name = det_model_name
        self.rec_model_name = rec_model_name
        self.enable_mkldnn = enable_mkldnn
        self.use_angle_cls = use_angle_cls
        self._ocr = None

    def _get_ocr_instance(self):
        """Lazy initialization of PaddleOCR instance with singleton reuse."""
        cache_key = f"{self.language}_{self.use_gpu}_{self.cpu_threads}_{self.det_model_name}_{self.rec_model_name}"
        if cache_key in OCREngine._shared_instances:
            self._ocr = OCREngine._shared_instances[cache_key]
            return self._ocr

        if self._ocr is None:
            logger.info(
                "Initializing PaddleOCR engine (det=%s, rec=%s, gpu=%s, cpu_threads=%d)...",
                self.det_model_name, self.rec_model_name, self.use_gpu, self.cpu_threads,
            )
            t_str = str(self.cpu_threads)
            os.environ["OMP_NUM_THREADS"] = t_str
            os.environ["MKL_NUM_THREADS"] = t_str
            os.environ["OPENBLAS_NUM_THREADS"] = t_str
            os.environ["FLAGS_enable_pir_api"] = "0"
            os.environ["PADDLE_DISABLE_ONEDNN"] = "0" if self.enable_mkldnn else "1"

            try:
                from paddleocr import PaddleOCR
                self._ocr = PaddleOCR(
                    lang=self.language,
                    device="gpu" if self.use_gpu else "cpu",
                    enable_mkldnn=self.enable_mkldnn,
                    cpu_threads=self.cpu_threads,
                    text_detection_model_name=self.det_model_name,
                    text_recognition_model_name=self.rec_model_name,
                    use_doc_orientation_classify=False,
                    use_doc_unwarping=False,
                    use_textline_orientation=self.use_angle_cls,
                )

                OCREngine._shared_instances[cache_key] = self._ocr
                OCREngine._shared_ocr = self._ocr
                logger.info("PaddleOCR engine (%s + %s) initialized successfully.", self.det_model_name, self.rec_model_name)
            except Exception as e:
                logger.error("Failed to initialize PaddleOCR (%s): %s", self.rec_model_name, e)
                raise
        return self._ocr

    def warm_up(self) -> None:
        """Eagerly load models and warm up inference with a tiny dummy image."""
        try:
            ocr = self._get_ocr_instance()
            dummy = np.zeros((64, 64, 3), dtype=np.uint8)
            ocr.ocr(dummy)
        except Exception as e:
            logger.warning("OCREngine warm_up warning: %s", e)

    def _load_image_safe(self, image_path: Path) -> np.ndarray:
        """
        Safely load image with PIL and OpenCV, capping dimensions to avoid memory bloat.
        Matches Finverge backend preprocessing.
        """
        with Image.open(image_path) as pil_img:
            rgb_img = pil_img.convert("RGB")
            arr = np.array(rgb_img)

        if cv2 is not None:
            img = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            h, w = img.shape[:2]
            max_dim = 1600
            min_dim = 64
            if h < min_dim or w < min_dim:
                scale = min_dim / min(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_CUBIC)
            elif h > max_dim or w > max_dim:
                scale = max_dim / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
            return np.ascontiguousarray(img)
        return arr

    def extract_text(self, image_path: Union[str, Path]) -> OCRResult:
        """
        Extract text from an image file in a single OCR pass using backend models.
        Never raises unhandled exceptions to caller.
        """
        img_path = Path(image_path)
        if not img_path.is_file():
            return OCRResult(
                text="",
                success=False,
                error=f"File not found: {img_path}",
                rec_model_name=self.rec_model_name,
            )

        t_load0 = time.perf_counter()
        try:
            with Image.open(img_path) as test_img:
                test_img.verify()
        except Exception as e:
            return OCRResult(
                text="",
                success=False,
                error=f"Corrupt or unreadable image file: {e}",
                rec_model_name=self.rec_model_name,
            )

        try:
            img_arr = self._load_image_safe(img_path)
        except Exception as e:
            return OCRResult(
                text="",
                success=False,
                error=f"Image load/conversion error: {e}",
                rec_model_name=self.rec_model_name,
            )
        file_load_sec = time.perf_counter() - t_load0

        t_init0 = time.perf_counter()
        try:
            ocr = self._get_ocr_instance()
        except Exception as e:
            return OCRResult(
                text="",
                success=False,
                error=f"OCR initialization error: {e}",
                file_load_time_sec=file_load_sec,
                rec_model_name=self.rec_model_name,
            )
        ocr_init_sec = time.perf_counter() - t_init0

        t_inf0 = time.perf_counter()
        try:
            raw_output = ocr.ocr(img_arr)
            ocr_inference_sec = time.perf_counter() - t_inf0
        except Exception as e:
            logger.error("PaddleOCR execution failed on '%s': %s", img_path.name, e)
            return OCRResult(
                text="",
                success=False,
                error=f"OCR execution error: {e}",
                file_load_time_sec=file_load_sec,
                ocr_init_time_sec=ocr_init_sec,
                rec_model_name=self.rec_model_name,
            )

        t_ext0 = time.perf_counter()
        extracted_lines: List[OCRLine] = []
        full_text_parts: List[str] = []
        conf_scores: List[float] = []

        if raw_output:
            pages = raw_output if isinstance(raw_output, list) else [raw_output]
            for page in pages:
                if not page:
                    continue

                # Format 1: Standard legacy list format [[box, (text, conf)], ...]
                if isinstance(page, list):
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

                # Format 2: PaddleX Dictionary output (v3.x / v6 models)
                elif isinstance(page, dict):
                    rec_texts = page.get("rec_texts", page.get("texts", []))
                    rec_scores = page.get("rec_scores", page.get("scores", []))
                    rec_polys = page.get("rec_polys", page.get("dt_polys", page.get("boxes", [])))
                    for i, txt in enumerate(rec_texts):
                        txt_str = str(txt).strip()
                        if not txt_str:
                            continue
                        score = float(rec_scores[i]) if i < len(rec_scores) else 1.0
                        poly = rec_polys[i] if i < len(rec_polys) else None
                        extracted_lines.append(OCRLine(text=txt_str, confidence=score, box=poly))
                        full_text_parts.append(txt_str)
                        conf_scores.append(score)

        combined_text = "\n".join(full_text_parts)
        mean_conf = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0
        ocr_extract_sec = time.perf_counter() - t_ext0

        return OCRResult(
            text=combined_text,
            lines=extracted_lines,
            mean_confidence=mean_conf,
            success=True,
            error=None,
            file_load_time_sec=file_load_sec,
            preprocessing_time_sec=0.0,
            ocr_init_time_sec=ocr_init_sec,
            ocr_inference_time_sec=ocr_inference_sec,
            ocr_extract_time_sec=ocr_extract_sec,
            crop_count=len(extracted_lines),
            rec_model_name=self.rec_model_name,
        )
