"""

Batch Processor for Large-Scale KYC Document Segregation.

Scans source directories, coordinates OCR, classification, and routing,

provides per-file fault isolation, generates summary reports and CSV logs.

"""

from __future__ import annotations

import csv

import datetime

import logging

import time

import os

from concurrent.futures import ProcessPoolExecutor

from dataclasses import dataclass, field

from pathlib import Path

from typing import Any, Callable, Dict, List, Optional, Set, Tuple

from kyc_sorter.detector.classifier import ClassificationResult, DocumentClassifier

from kyc_sorter.detector.ocr import OCREngine, OCRResult

from kyc_sorter.sorter.file_router import FileRouter, RouteDecision

from kyc_sorter.utils.config_loader import Config

logger = logging.getLogger(__name__)

# Process-level singletons for worker processes (initialized once per process)

_WORKER_OCR_ENGINE: Optional[OCREngine] = None

_WORKER_CLASSIFIER: Optional[DocumentClassifier] = None

def compute_adaptive_workers(

    configured_max: Optional[int] = None,

    total_cores: Optional[int] = None,

    adaptive_enabled: bool = True,

    target_cpu_percent: float = 80.0,

) -> int:

    """

    Determine safe worker process count adaptively based on available CPU cores.

    Leaves headroom for the OS and desktop applications:

      - 1-2 cores: 1 worker

      - 3-4 cores: 2 workers

      - 5-7 cores: 3 workers

      - 8-11 cores: min(configured_max, cores // 2)

      - 12+ cores: min(configured_max, max(2, int(cores * (target_cpu_percent / 100.0) * 0.7)))

    """

    cores = total_cores if (total_cores is not None and total_cores >= 1) else (os.cpu_count() or 1)

    limit = configured_max if (configured_max is not None and configured_max >= 1) else 6

    if limit <= 1 or not adaptive_enabled:

        return max(1, limit)

    if cores <= 2:

        adaptive = 1

    elif cores <= 4:

        adaptive = 2

    elif cores <= 7:

        adaptive = 3

    elif cores <= 11:

        adaptive = max(2, cores // 2)

    else:

        adaptive = max(2, min(limit, int(cores * (target_cpu_percent / 100.0) * 0.7)))

    return max(1, min(adaptive, limit))

def _worker_process_init(

    confidence_threshold: float,

    language: str,

    cpu_threads: int,

    multiple_document_score_margin: float = 0.10,

    det_model_name: Optional[str] = "PP-OCRv5_mobile_det",

    rec_model_name: Optional[str] = "en_PP-OCRv5_mobile_rec",

    enable_mkldnn: bool = False,

    use_angle_classification: bool = False,

):

    """

    Worker process initializer.

    Executed ONCE per worker process when spawned.

    Instantiates and pre-initializes OCREngine and DocumentClassifier as process singletons.

    """

    global _WORKER_OCR_ENGINE, _WORKER_CLASSIFIER

    # Configure safety env vars for worker process

    os.environ["FLAGS_enable_pir_api"] = "0"

    os.environ.pop("PADDLE_DISABLE_ONEDNN", None)

    os.environ["PADDLEX_DISABLE_AUTO_INIT"] = "True"

    os.environ["PADDLEX_LOG_LEVEL"] = "CRITICAL"

    t_str = str(max(1, cpu_threads))

    os.environ["OMP_NUM_THREADS"] = t_str

    os.environ["MKL_NUM_THREADS"] = t_str

    os.environ["OPENBLAS_NUM_THREADS"] = t_str

    try:

        _WORKER_OCR_ENGINE = OCREngine(

            language=language,

            use_gpu=False,

            cpu_threads=cpu_threads,

            det_model_name=det_model_name or "PP-OCRv5_mobile_det",

            rec_model_name=rec_model_name or "en_PP-OCRv5_mobile_rec",

            enable_mkldnn=enable_mkldnn,

            use_angle_cls=use_angle_classification,

        )

        _WORKER_OCR_ENGINE.warm_up()

        _WORKER_CLASSIFIER = DocumentClassifier(

            confidence_threshold=confidence_threshold,

            multiple_document_score_margin=multiple_document_score_margin,

        )

    except Exception as e:

        logger.error("Failed to initialize OCREngine or DocumentClassifier in worker process: %s", e)

def _worker_process_task(file_path_str: str) -> Dict[str, Any]:

    """

    Worker task: runs single-pass OCR and classification.

    Reuses backend models PP-OCRv5_mobile_det and en_PP-OCRv5_mobile_rec.

    """

    global _WORKER_OCR_ENGINE, _WORKER_CLASSIFIER

    t0 = time.time()

    try:

        file_path = Path(file_path_str)

        if _WORKER_OCR_ENGINE is None:

            _WORKER_OCR_ENGINE = OCREngine()

            _WORKER_OCR_ENGINE.warm_up()

        if _WORKER_CLASSIFIER is None:

            _WORKER_CLASSIFIER = DocumentClassifier()

        ocr_res = _WORKER_OCR_ENGINE.extract_text(file_path)

        class_res = _WORKER_CLASSIFIER.classify_text(ocr_res.text, ocr_res.lines)

        duration = time.time() - t0

        return {

            "source_path": file_path_str,

            "document_type": class_res.document_type,

            "confidence": class_res.confidence,

            "status": class_res.status,

            "matched_template": class_res.matched_template,

            "error": ocr_res.error or class_res.error,

            "processing_time_sec": duration,

            "rejection_category": class_res.rejection_category,

            "timing_dict": {

                "file_load_sec": getattr(ocr_res, "file_load_time_sec", 0.0),

                "preprocessing_sec": getattr(ocr_res, "preprocessing_time_sec", 0.0),

                "ocr_init_sec": getattr(ocr_res, "init_time_sec", 0.0),

                "ocr_inference_sec": getattr(ocr_res, "inference_time_sec", 0.0),

                "ocr_processing_sec": getattr(ocr_res, "postprocessing_time_sec", 0.0),

                "classification_sec": getattr(class_res, "classification_time_sec", 0.0),

                "confidence_sec": getattr(class_res, "confidence_time_sec", 0.0),

                "multi_doc_detection_sec": getattr(class_res, "multi_doc_time_sec", 0.0),

            },

            "ocr_text": ocr_res.text,

            "matched_signals": class_res.matched_signals,

        }

    except Exception as e:

        duration = time.time() - t0

        return {

            "source_path": file_path_str,

            "document_type": "UNKNOWN",

            "confidence": 0.0,

            "status": "FAILED",

            "matched_template": None,

            "error": str(e),

            "processing_time_sec": duration,

            "rejection_category": "ERROR",

            "timing_dict": {},

            "ocr_text": "",

            "matched_signals": {"error": str(e)},

        }

@dataclass

class DocumentProfileTiming:

    file_load_sec: float = 0.0

    preprocessing_sec: float = 0.0

    ocr_init_sec: float = 0.0

    ocr_inference_sec: float = 0.0

    ocr_processing_sec: float = 0.0

    classification_sec: float = 0.0

    confidence_sec: float = 0.0

    multi_doc_detection_sec: float = 0.0

    routing_sec: float = 0.0

    file_operation_sec: float = 0.0

    logging_sec: float = 0.0

    total_sec: float = 0.0

    def format_profiling_block(self, filename: str) -> str:

        lines = [

            "-" * 50,

            f"FILE: {filename}",

            "",

            f"OCR pass:               Single-pass (PP-OCRv5_mobile)",

            f"File load:              {self.file_load_sec:.2f} sec",

            f"Preprocessing:          {self.preprocessing_sec:.2f} sec",

            f"OCR inference:          {self.ocr_inference_sec:.2f} sec",

            f"OCR processing:         {self.ocr_processing_sec:.2f} sec",

            f"Classification:         {self.classification_sec:.2f} sec",

            f"Confidence:             {self.confidence_sec:.2f} sec",

            f"Multi-doc detection:    {self.multi_doc_detection_sec:.2f} sec",

            f"Routing:                {self.routing_sec:.2f} sec",

            f"File operation:         {self.file_operation_sec:.2f} sec",

            f"Logging:                {self.logging_sec:.2f} sec",

            "",

            f"TOTAL:                  {self.total_sec:.2f} sec",

            "-" * 50,

        ]

        return "\n".join(lines)

@dataclass

class ProcessedItem:

    timestamp: str

    source_file: str

    source_path: str

    document_type: str

    confidence: float

    status: str

    matched_template: Optional[str]

    destination_path: str

    operation: str

    dry_run: bool

    error: Optional[str] = None

    processing_time_sec: float = 0.0

    timing: Optional[DocumentProfileTiming] = None

    rejection_category: str = "KYC"

@dataclass

class BatchSummary:

    total_files: int = 0

    processed: int = 0

    failed: int = 0

    skipped_unsupported: int = 0

    counts_by_type: Dict[str, int] = field(default_factory=dict)

    items: List[ProcessedItem] = field(default_factory=list)

    csv_report_path: Optional[Path] = None

    elapsed_time_sec: float = 0.0

    avg_time_per_image: float = 0.0

    min_time_per_image: float = 0.0

    max_time_per_image: float = 0.0

    avg_ocr_sec: float = 0.0

    avg_classification_sec: float = 0.0

    avg_file_op_sec: float = 0.0

    avg_other_sec: float = 0.0

    correctly_classified_kyc: int = 0

    classified_non_kyc: int = 0

    classified_unknown: int = 0

    rejected_low_confidence: int = 0

    identified_multi_doc: int = 0

    failed_errors: int = 0

    max_active_workers: int = 1

    avg_cpu_percent: float = 0.0

    peak_cpu_percent: float = 0.0

    peak_memory_mb: float = 0.0

    kyc_routed_count: int = 0

    non_kyc_routed_count: int = 0

    def print_summary(self):

        """Display clean terminal performance summary matching Section 5 requirements."""

        print("\n" + "=" * 60)

        print(f"PERFORMANCE & SEGREGATION SUMMARY (workers={self.max_active_workers})")

        print("=" * 60)

        print(f"Total documents: {self.total_files}")

        print("")

        print("Average:")

        print(f"- OCR:            {self.avg_ocr_sec:.2f} sec")

        print(f"- Classification: {self.avg_classification_sec:.2f} sec")

        print(f"- File operation: {self.avg_file_op_sec:.2f} sec")

        print(f"- Other:          {self.avg_other_sec:.2f} sec")

        print(f"- Total/document: {self.avg_time_per_image:.2f} sec")

        print("")

        print(f"Minimum/document: {self.min_time_per_image:.2f} sec")

        print(f"Maximum/document: {self.max_time_per_image:.2f} sec")

        print("")

        print(f"Total execution time: {self.elapsed_time_sec:.2f} sec")

        print("-" * 60)

        print("-" * 60)

        print("Document Classification & Routing Breakdown:")

        print(f"- correctly classified as KYC        : {self.correctly_classified_kyc}")

        print(f"- classified as NON_KYC              : {self.classified_non_kyc}")

        print(f"- classified as UNKNOWN              : {self.classified_unknown}")

        print(f"- rejected due to low confidence     : {self.rejected_low_confidence}")

        print(f"- identified as multiple-document    : {self.identified_multi_doc}")

        print(f"- failed due to errors               : {self.failed_errors}")

        if self.skipped_unsupported > 0:

            print(f"- unsupported files (ignored)        : {self.skipped_unsupported}")

        print("-" * 60)

        print("System Resource Statistics:")

        print(f"- Active Workers                     : {self.max_active_workers}")

        print(f"- Avg CPU Usage                      : {self.avg_cpu_percent:.1f}%")

        print(f"- Peak CPU Usage                     : {self.peak_cpu_percent:.1f}%")

        if self.peak_memory_mb > 0:

            print(f"- Peak Memory (RSS)                  : {self.peak_memory_mb:.1f} MB")

        print("-" * 60)

        print("KYC Document Sub-types:")

        for doc_type, count in sorted(self.counts_by_type.items()):

            if doc_type != "UNKNOWN":

                print(f"  {doc_type:<32}: {count}")

        print("=" * 60 + "\n")

TEMPORARY_EXTENSIONS = frozenset({

    ".tmp", ".part", ".crdownload", ".download", ".partial",

    ".swp", ".lock", ".temp", ".pending", ".uploading"

})

class BatchProcessor:

    """

    Coordinates end-to-end processing of KYC documents from source to destination.

    """

    def __init__(

        self,

        config: Config,

        ocr_engine: Optional[OCREngine] = None,

        classifier: Optional[DocumentClassifier] = None,

        router: Optional[FileRouter] = None,

    ):

        self.config = config

        self.ocr_engine = ocr_engine or OCREngine(

            language=config.ocr.language,

            use_gpu=config.ocr.use_gpu,

            cpu_threads=config.ocr.cpu_threads,

            det_model_name=getattr(config.ocr, "text_detection_model_name", "PP-OCRv5_mobile_det"),

            rec_model_name=getattr(config.ocr, "text_recognition_model_name", "en_PP-OCRv5_mobile_rec"),

            enable_mkldnn=getattr(config.ocr, "enable_mkldnn", False),

            use_angle_cls=getattr(config.ocr, "use_angle_classification", False),

        )

        self.classifier = classifier or DocumentClassifier(

            confidence_threshold=config.processing.confidence_threshold,

            multiple_document_score_margin=getattr(config.processing, "multiple_document_score_margin", 0.10),

        )

        self.router = router or FileRouter(

            output_directory=config.output.resolved_path,

            folders_mapping=config.folders,

            suffixes_mapping=config.suffixes,

            kyc_folder=config.kyc_folder,

            unknown_folder=config.unknown_folder,

            unknown_suffix=config.unknown_suffix,

            operation=config.output.operation,

            create_date_folder=config.output.create_date_folder,

            create_subfolders=False,

            date_format=config.output.date_format,

            dry_run=config.processing.dry_run,

        )

        self._abort_requested = False

    def abort(self):

        """Signal batch processor to safely stop processing further files."""

        self._abort_requested = True

        logger.warning("Batch processing abort requested by user.")

    def scan_source_files(self, source_dir: Optional[Path] = None) -> Tuple[List[Path], List[Path]]:

        """

        Scan directory for supported image files, returning (supported_files, unsupported_files).

        Excludes temporary and partial files (.tmp, .part, .crdownload).

        """

        src = Path(source_dir) if source_dir else self.config.input.resolved_path

        supported: List[Path] = []

        unsupported: List[Path] = []

        if not src.is_dir():

            logger.warning("Source directory does not exist: %s", src)

            return supported, unsupported

        valid_exts = self.config.processing.supported_extensions

        for item in sorted(src.iterdir()):

            if item.is_file():

                ext = item.suffix.lower()

                # Skip temporary / partial / hidden files

                if ext in TEMPORARY_EXTENSIONS or item.name.startswith(".") or item.name.startswith("~"):

                    continue

                if ext in valid_exts:

                    supported.append(item)

                else:

                    unsupported.append(item)

        return supported, unsupported

    def process_file(

        self,

        file_path: Path,

        processing_date: Optional[datetime.date] = None,

    ) -> ProcessedItem:

        """

        Process a single document through single-pass OCR, classification, and routing.

        Guaranteed not to raise uncaught exceptions.

        """

        start_time = time.perf_counter()

        ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Ignore non-supported file formats completely (do not OCR, classify, copy, move, or rename)

        ext = file_path.suffix.lower()

        if ext not in self.config.processing.supported_extensions:

            logger.info("Ignoring non-supported file '%s' (only %s supported).", file_path.name, ", ".join(sorted(self.config.processing.supported_extensions)))

            return ProcessedItem(

                timestamp=ts_str,

                source_file=file_path.name,

                source_path=str(file_path.resolve()),

                document_type="IGNORED",

                confidence=0.0,

                status="IGNORED",

                matched_template=None,

                destination_path="",

                operation=self.config.output.operation,

                dry_run=self.config.processing.dry_run,

                error=f"Ignored non-supported format: {file_path.suffix}",

                processing_time_sec=0.0,

                rejection_category="IGNORED",

            )

        try:

            # 1. Single-pass OCR & Classification

            t_ocr0 = time.perf_counter()

            ocr_res = self.ocr_engine.extract_text(file_path)

            ocr_elapsed = time.perf_counter() - t_ocr0

            t_class0 = time.perf_counter()

            class_res = self.classifier.classify_text(ocr_res.text, ocr_res.lines)

            class_elapsed = time.perf_counter() - t_class0

            # Determine rejection reason

            is_kyc = class_res.is_kyc(self.config.processing.confidence_threshold)

            reason_non_kyc = None

            if not is_kyc:

                raw_sig = class_res.matched_signals or {}

                if class_res.rejection_category == "MULTI_DOC":

                    reason_non_kyc = f"Multi-document detected ({raw_sig.get('reason', 'Multiple documents')})"

                elif class_res.rejection_category == "LOW_CONFIDENCE":

                    conf_str = f"{class_res.confidence:.2f}"

                    if float(conf_str) >= self.config.processing.confidence_threshold:

                        conf_str = f"{class_res.confidence:.4f}"

                        if float(conf_str) >= self.config.processing.confidence_threshold:

                            conf_str = str(class_res.confidence)

                    reason_non_kyc = f"Low confidence ({conf_str} < {self.config.processing.confidence_threshold:.2f})"

                elif class_res.rejection_category in ("NON_KYC", "UNKNOWN"):

                    reason_non_kyc = "No KYC document template matched"

                elif class_res.rejection_category == "ERROR":

                    reason_non_kyc = f"OCR/Processing error: {class_res.error or ocr_res.error}"

                else:

                    reason_non_kyc = class_res.rejection_category

            logger.info(

                "[CLASSIFICATION RECEIVED] [%s] '%s' | %s | conf=%.2f | category=%s%s (ocr: %.3fs, class: %.3fs)",

                ts_str,

                file_path.name,

                class_res.document_type,

                class_res.confidence,

                class_res.rejection_category,

                f" | reason='{reason_non_kyc}'" if reason_non_kyc else "",

                ocr_elapsed,

                class_elapsed,

            )

            # 2. File Routing based on authoritative classification

            generate_companions = getattr(self.config.output, "generate_companion_files", True)

            decision = self.router.route(

                file_path,

                class_res.document_type,

                processing_date=processing_date,

                ocr_text=ocr_res.text,

                rejection_reason=reason_non_kyc,

                classification_status=class_res.status,

                confidence=class_res.confidence,

                rejection_category=class_res.rejection_category,

                generate_companions=generate_companions,

            )

            # 3. Logging & Profiling

            t_log0 = time.perf_counter()

            duration = time.perf_counter() - start_time

            logger.info(

                "[%s] %s | %s | conf=%.2f | total=%.2fs | dest=%s | status=%s",

                ts_str,

                file_path.name,

                class_res.document_type,

                class_res.confidence,

                duration,

                decision.destination_path,

                class_res.status,

            )

            t_log = time.perf_counter() - t_log0

            timing = DocumentProfileTiming(

                file_load_sec=getattr(ocr_res, "file_load_time_sec", 0.0),

                preprocessing_sec=getattr(ocr_res, "preprocessing_time_sec", 0.0),

                ocr_init_sec=getattr(ocr_res, "init_time_sec", 0.0),

                ocr_inference_sec=getattr(ocr_res, "inference_time_sec", 0.0),

                ocr_processing_sec=getattr(ocr_res, "postprocessing_time_sec", 0.0),

                classification_sec=getattr(class_res, "classification_time_sec", 0.0),

                confidence_sec=getattr(class_res, "confidence_time_sec", 0.0),

                multi_doc_detection_sec=getattr(class_res, "multi_doc_time_sec", 0.0),

                routing_sec=getattr(decision, "routing_time_sec", 0.0),

                file_operation_sec=getattr(decision, "file_op_time_sec", 0.0),

                logging_sec=t_log,

                total_sec=duration,

            )

            print("\n" + timing.format_profiling_block(file_path.name) + "\n", flush=True)

            return ProcessedItem(

                timestamp=ts_str,

                source_file=file_path.name,

                source_path=str(file_path.resolve()),

                document_type=class_res.document_type,

                confidence=class_res.confidence,

                status=class_res.status,

                matched_template=class_res.matched_template,

                destination_path=str(decision.destination_path),

                operation=self.config.output.operation,

                dry_run=self.config.processing.dry_run,

                error=class_res.error or ocr_res.error or decision.error,

                processing_time_sec=duration,

                timing=timing,

                rejection_category=class_res.rejection_category,

            )

        except Exception as e:

            duration = time.perf_counter() - start_time

            logger.error("Unexpected error processing '%s': %s", file_path.name, e, exc_info=True)

            # Route to unknown on unexpected failure

            try:

                generate_companions = getattr(self.config.output, "generate_companion_files", True)

                decision = self.router.route(

                    file_path,

                    "UNKNOWN",

                    processing_date=processing_date,

                    ocr_text="",

                    rejection_reason=f"Processing exception: {str(e)}",

                    classification_status="FAILED",

                    confidence=0.0,

                    rejection_category="ERROR",

                    generate_companions=generate_companions,

                )

                dest_str = str(decision.destination_path)

            except Exception:

                dest_str = str(self.config.output.resolved_path / self.config.unknown_folder / file_path.name)

            timing = DocumentProfileTiming(total_sec=duration)

            print("\n" + timing.format_profiling_block(file_path.name) + "\n", flush=True)

            return ProcessedItem(

                timestamp=ts_str,

                source_file=file_path.name,

                source_path=str(file_path.resolve()),

                document_type="UNKNOWN",

                confidence=0.0,

                status="FAILED",

                matched_template=None,

                destination_path=dest_str,

                operation=self.config.output.operation,

                dry_run=self.config.processing.dry_run,

                error=str(e),

                processing_time_sec=duration,

                timing=timing,

                rejection_category="ERROR",

            )

    def process_batch(

        self,

        source_dir: Optional[Path] = None,

        progress_callback: Optional[Callable[[int, int, ProcessedItem], None]] = None,

        limit: Optional[int] = None,

    ) -> BatchSummary:

        """

        Execute full batch processing across all files in source directory.

        """

        batch_start = time.time()

        supported_files, unsupported_files = self.scan_source_files(source_dir)


        # No file-count limit: process every supported file found in the source directory.

        total_count = len(supported_files)

        logger.info("Found %d supported files to process (%d unsupported skipped).", total_count, len(unsupported_files))

        summary = BatchSummary(

            total_files=total_count,

            skipped_unsupported=len(unsupported_files),

        )

        if total_count == 0:

            summary.elapsed_time_sec = time.time() - batch_start

            return summary

        # Determine adaptive worker count

        configured_max = getattr(self.config.processing, "max_workers", 6)

        cpu_limit = getattr(self.config.processing, "cpu_limit_percent", 80.0)

        adaptive = getattr(self.config.processing, "adaptive_workers", True)

        score_margin = getattr(self.config.processing, "multiple_document_score_margin", 0.10)

        effective_workers = compute_adaptive_workers(

            configured_max,

            adaptive_enabled=adaptive,

            target_cpu_percent=cpu_limit,

        )

        num_workers = min(effective_workers, total_count)

        cpu_samples: List[float] = []

        peak_memory_mb: float = 0.0

        try:

            import psutil

            current_proc = psutil.Process()

            psutil.cpu_percent(interval=None)  # Prime initial measurement

        except Exception:

            current_proc = None

        if num_workers <= 1:

            logger.info("Processing %d files sequentially (workers=1)...", total_count)

            for i, fpath in enumerate(supported_files, start=1):

                if self._abort_requested:

                    logger.warning("Batch processing aborted after %d files.", summary.processed)

                    break

                item = self.process_file(fpath)

                summary.items.append(item)

                summary.processed += 1

                if current_proc is not None:

                    try:

                        c_pct = psutil.cpu_percent(interval=None)

                        if c_pct > 0.0:

                            cpu_samples.append(c_pct)

                        mem_rss = current_proc.memory_info().rss / (1024 * 1024)

                        if mem_rss > peak_memory_mb:

                            peak_memory_mb = mem_rss

                    except Exception:

                        pass

                if item.status == "FAILED":

                    summary.failed += 1

                doc_type = item.document_type

                summary.counts_by_type[doc_type] = summary.counts_by_type.get(doc_type, 0) + 1

                if progress_callback:

                    progress_callback(i, total_count, item)

        else:

            logger.info(

                "Processing %d files concurrently using %d worker processes (adaptive limit from max %d, target CPU ceiling %.0f%% on %d CPU cores)...",

                total_count,

                num_workers,

                configured_max,

                cpu_limit,

                os.cpu_count() or 1,

            )

            import concurrent.futures

            import threading

            file_path_strs = [str(f.resolve()) for f in supported_files]

            pending_files = list(file_path_strs)

            in_flight: Dict[concurrent.futures.Future, str] = {}

            completed_count = 0

            monitor_stop = threading.Event()

            def _resource_monitor():
                nonlocal peak_memory_mb

                if current_proc is not None:
                    try:
                        psutil.cpu_percent(interval=None)
                    except Exception:
                        pass

                while not monitor_stop.wait(1.0):
                    if current_proc is None:
                        break
                    try:
                        c_val = psutil.cpu_percent(interval=None)
                        if c_val > 0.0:
                            cpu_samples.append(c_val)
                        if c_val >= cpu_limit:
                            logger.warning(
                                "CPU usage is %.1f%% (configured target %.1f%%). "
                                "New task submission will pause until usage decreases.",
                                c_val,
                                cpu_limit,
                            )

                        total_rss = current_proc.memory_info().rss
                        for child in current_proc.children(recursive=True):
                            try:
                                total_rss += child.memory_info().rss
                            except Exception:
                                pass
                        mem_rss_mb = total_rss / (1024 * 1024)
                        if mem_rss_mb > peak_memory_mb:
                            peak_memory_mb = mem_rss_mb
                    except Exception as e:
                        logger.debug("Resource monitoring error: %s", e)

            monitor_thread = threading.Thread(target=_resource_monitor, daemon=True)

            monitor_thread.start()

            try:

                with ProcessPoolExecutor(

                    max_workers=num_workers,

                    initializer=_worker_process_init,

                    initargs=(

                        self.config.processing.confidence_threshold,

                        self.config.ocr.language,

                        self.config.ocr.cpu_threads,

                        score_margin,

                        getattr(self.config.ocr, "text_detection_model_name", "PP-OCRv5_mobile_det"),

                        getattr(self.config.ocr, "text_recognition_model_name", "en_PP-OCRv5_mobile_rec"),

                        getattr(self.config.ocr, "enable_mkldnn", False),

                        getattr(self.config.ocr, "use_angle_classification", False),

                    ),

                ) as executor:

                    while (pending_files or in_flight) and not self._abort_requested:

                        # Dynamic Adaptive Worker Scheduling & CPU Throttling

                        latest_cpu = cpu_samples[-1] if cpu_samples else 0.0

                        is_overloaded = adaptive and (latest_cpu >= cpu_limit) and (len(in_flight) >= 1)

                        if not is_overloaded:

                            while pending_files and (len(in_flight) < num_workers):

                                next_file_str = pending_files.pop(0)

                                fut = executor.submit(_worker_process_task, next_file_str)

                                in_flight[fut] = next_file_str

                                if adaptive and len(in_flight) >= 2:

                                    try:

                                        if psutil.cpu_percent(interval=None) >= cpu_limit:

                                            break

                                    except Exception:

                                        pass

                        # Process completed task futures

                        if in_flight:

                            done, _ = concurrent.futures.wait(

                                in_flight.keys(),

                                timeout=0.15 if is_overloaded else 0.05,

                                return_when=concurrent.futures.FIRST_COMPLETED,

                            )

                            for fut in done:

                                in_flight.pop(fut, None)

                                completed_count += 1

                                raw_res = fut.result()

                                fpath = Path(raw_res["source_path"])

                                doc_type = raw_res["document_type"]

                                status = raw_res["status"]

                                error = raw_res["error"]

                                ocr_text = raw_res.get("ocr_text", "")

                                rej_cat = raw_res.get("rejection_category", "KYC")

                                conf_val = raw_res.get("confidence", 0.0)

                                sigs = raw_res.get("matched_signals", {}) or {}

                                is_kyc = (

                                    doc_type != "UNKNOWN"

                                    and conf_val >= self.config.processing.confidence_threshold

                                    and rej_cat == "KYC"

                                )

                                reason_non_kyc = None

                                if not is_kyc:

                                    if rej_cat == "MULTI_DOC":

                                        reason_non_kyc = f"Multi-document detected ({sigs.get('reason', 'Multiple documents')})"

                                    elif rej_cat == "LOW_CONFIDENCE":

                                        conf_str = f"{conf_val:.2f}"

                                        if float(conf_str) >= self.config.processing.confidence_threshold:

                                            conf_str = f"{conf_val:.4f}"

                                            if float(conf_str) >= self.config.processing.confidence_threshold:

                                                conf_str = str(conf_val)

                                        reason_non_kyc = f"Low confidence ({conf_str} < {self.config.processing.confidence_threshold:.2f})"

                                    elif rej_cat in ("NON_KYC", "UNKNOWN"):

                                        reason_non_kyc = "No KYC document template matched"

                                    elif rej_cat == "ERROR":

                                        reason_non_kyc = f"OCR/Processing error: {error}"

                                    else:

                                        reason_non_kyc = rej_cat

                                ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                                logger.info(

                                    "[CLASSIFICATION RECEIVED] [%s] '%s' | %s | conf=%.2f | category=%s%s",

                                    ts_str,

                                    fpath.name,

                                    doc_type,

                                    conf_val,

                                    rej_cat,

                                    f" | reason='{reason_non_kyc}'" if reason_non_kyc else "",

                                )

                                # Execute safe sequential routing & collision handling on main process

                                generate_companions = getattr(self.config.output, "generate_companion_files", True)

                                decision = self.router.route(

                                    fpath,

                                    doc_type,

                                    ocr_text=ocr_text,

                                    rejection_reason=reason_non_kyc,

                                    classification_status=status,

                                    confidence=conf_val,

                                    rejection_category=rej_cat,

                                    generate_companions=generate_companions,

                                )

                                td = raw_res.get("timing_dict")

                                if td:

                                    timing = DocumentProfileTiming(

                                        file_load_sec=td.get("file_load_sec", 0.0),

                                        preprocessing_sec=td.get("preprocessing_sec", 0.0),

                                        ocr_init_sec=td.get("ocr_init_sec", 0.0),

                                        ocr_inference_sec=td.get("ocr_inference_sec", 0.0),

                                        ocr_processing_sec=td.get("ocr_processing_sec", 0.0),

                                        classification_sec=td.get("classification_sec", 0.0),

                                        confidence_sec=td.get("confidence_sec", 0.0),

                                        multi_doc_detection_sec=td.get("multi_doc_detection_sec", 0.0),

                                        routing_sec=decision.routing_time_sec,

                                        file_operation_sec=decision.file_op_time_sec,

                                        total_sec=raw_res["processing_time_sec"],

                                    )

                                    print("\n" + timing.format_profiling_block(fpath.name) + "\n", flush=True)

                                else:

                                    timing = DocumentProfileTiming(

                                        total_sec=raw_res["processing_time_sec"],

                                    )

                                item = ProcessedItem(

                                    timestamp=ts_str,

                                    source_file=fpath.name,

                                    source_path=str(fpath),

                                    document_type=doc_type,

                                    confidence=raw_res["confidence"],

                                    status=status,

                                    matched_template=raw_res["matched_template"],

                                    destination_path=str(decision.destination_path),

                                    operation=self.config.output.operation,

                                    dry_run=self.config.processing.dry_run,

                                    error=error or decision.error,

                                    processing_time_sec=raw_res["processing_time_sec"],

                                    timing=timing,

                                    rejection_category=raw_res.get("rejection_category", "KYC"),

                                )

                                summary.items.append(item)

                                summary.processed += 1

                                if item.status == "FAILED":

                                    summary.failed += 1

                                summary.counts_by_type[doc_type] = summary.counts_by_type.get(doc_type, 0) + 1

                                logger.info(

                                    "[%s] %s | %s | conf=%.2f | total=%.2fs | dest=%s | status=%s",

                                    item.timestamp,

                                    item.source_file,

                                    item.document_type,

                                    item.confidence,

                                    item.processing_time_sec,

                                    item.destination_path,

                                    item.status,

                                )

                                if progress_callback:

                                    progress_callback(completed_count, total_count, item)

                        elif pending_files and is_overloaded:

                            time.sleep(0.1)

            except KeyboardInterrupt:

                logger.warning("\nBatch processing interrupted by user (Ctrl+C). Terminating worker processes cleanly...")

            finally:

                monitor_stop.set()

                monitor_thread.join(timeout=2.0)

        summary.elapsed_time_sec = time.time() - batch_start

        summary.avg_time_per_image = summary.elapsed_time_sec / max(1, summary.processed)

        summary.max_active_workers = num_workers

        summary.avg_cpu_percent = (sum(cpu_samples) / len(cpu_samples)) if cpu_samples else 0.0

        summary.peak_cpu_percent = max(cpu_samples) if cpu_samples else 0.0

        summary.peak_memory_mb = peak_memory_mb

        summary.kyc_routed_count = sum(v for k, v in summary.counts_by_type.items() if k != "UNKNOWN")

        summary.non_kyc_routed_count = summary.counts_by_type.get("UNKNOWN", 0)

        # Performance summary metrics per Section 5

        valid_items = [it for it in summary.items if it.status != "IGNORED"]

        times = [it.processing_time_sec for it in valid_items]

        summary.min_time_per_image = min(times) if times else 0.0

        summary.max_time_per_image = max(times) if times else 0.0

        timings = [it.timing for it in valid_items if it.timing is not None]

        n_t = len(timings)

        if n_t > 0:

            summary.avg_ocr_sec = sum(t.file_load_sec + t.preprocessing_sec + t.ocr_inference_sec + t.ocr_processing_sec for t in timings) / n_t

            summary.avg_classification_sec = sum(t.classification_sec + t.confidence_sec + t.multi_doc_detection_sec for t in timings) / n_t

            summary.avg_file_op_sec = sum(t.routing_sec + t.file_operation_sec for t in timings) / n_t

            summary.avg_other_sec = sum(t.logging_sec for t in timings) / n_t

        summary.correctly_classified_kyc = sum(1 for it in valid_items if it.status == "IDENTIFIED" and it.document_type != "UNKNOWN")

        summary.classified_non_kyc = sum(1 for it in valid_items if it.document_type == "UNKNOWN")

        summary.classified_unknown = sum(1 for it in valid_items if it.document_type == "UNKNOWN")

        summary.rejected_low_confidence = sum(1 for it in valid_items if getattr(it, "rejection_category", "") == "LOW_CONFIDENCE")

        summary.identified_multi_doc = sum(1 for it in valid_items if getattr(it, "rejection_category", "") == "MULTI_DOC")

        summary.failed_errors = sum(1 for it in valid_items if it.status == "FAILED" or getattr(it, "rejection_category", "") == "ERROR")

        # CSV report generation is completely disabled per requirements

        summary.csv_report_path = None

        return summary

    def _write_csv_report(self, items: List[ProcessedItem]) -> Optional[Path]:

        """CSV report generation is completely disabled per requirements."""

        return None
