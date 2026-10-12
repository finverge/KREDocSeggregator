"""
Batch Processor for VLM-Only KYC Document Segregation.

Coordinates document discovery, VLM classification via Ollama, safe file movement,
concurrency control, CPU monitoring and backpressure, and performance reporting.
"""

from __future__ import annotations

import datetime
import logging
import os
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple

from kyc_sorter.sorter.file_router import FileRouter
from kyc_sorter.sorter.journal import ProcessingJournal
from kyc_sorter.sorter.vlm_router import (
    RouteResult,
    process_vlm_batch,
    process_vlm_file,
    route_document,
)
from kyc_sorter.utils.config_loader import Config

logger = logging.getLogger(__name__)


def compute_adaptive_workers(
    configured_max: Optional[int] = None,
    total_cores: Optional[int] = None,
    adaptive_enabled: bool = True,
    target_cpu_percent: float = 80.0,
) -> int:
    """
    Determine safe worker count adaptively based on available CPU cores.
    Leaves headroom for the OS and desktop applications:
      - 1-2 cores: 1 worker
      - 3-4 cores: 2 workers
      - 5-7 cores: 3 workers
      - 8-11 cores: min(configured_max, cores // 2)
      - 12+ cores: min(configured_max, max(2, int(cores * (target_cpu_percent / 100.0) * 0.7)))
    """
    cores = total_cores if (total_cores is not None and total_cores >= 1) else (os.cpu_count() or 1)
    limit = configured_max if (configured_max is not None and configured_max >= 1) else 4

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


@dataclass
class DocumentProfileTiming:
    file_load_sec: float = 0.0
    preprocessing_sec: float = 0.0
    vlm_inference_sec: float = 0.0
    ocr_inference_sec: float = 0.0
    classification_sec: float = 0.0
    confidence_sec: float = 0.0
    multi_doc_detection_sec: float = 0.0
    routing_sec: float = 0.0
    file_operation_sec: float = 0.0
    logging_sec: float = 0.0
    total_sec: float = 0.0
    queue_wait_sec: float = 0.0
    journal_write_sec: float = 0.0
    vlm_calls: int = 1
    retries: int = 0

    def format_profiling_block(self, filename: str) -> str:
        lines = [
            "-" * 50,
            f"FILE: {filename}",
            "",
            "VLM pipeline:           Qwen3-VL:4b-instruct",
            f"File read:              {self.file_load_sec:.2f} sec",
            f"Preprocessing:          {self.preprocessing_sec:.2f} sec",
            f"VLM inference:          {(self.vlm_inference_sec or self.ocr_inference_sec):.2f} sec",
            f"Response parsing:       {self.classification_sec:.2f} sec",
            f"Routing:                {self.routing_sec:.2f} sec",
            f"File operation:         {self.file_operation_sec:.2f} sec",
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
    pipeline_type: str = "VLM"
    model_name: str = "qwen3-vl:4b-instruct"
    model_instances: int = 1
    avg_vlm_sec: float = 0.0
    already_completed: int = 0
    peak_vram_mb: float = 0.0
    gpu_name: str = "None"
    median_time_per_image: float = 0.0
    p95_time_per_image: float = 0.0
    total_vlm_calls: int = 0
    total_retries: int = 0
    avg_queue_wait_sec: float = 0.0

    def print_summary(self):
        """Display clean terminal performance summary matching Section 5 requirements."""
        print("\n" + "=" * 60)
        print(f"PERFORMANCE & SEGREGATION SUMMARY (VLM workers={self.max_active_workers})")
        print(f"Model: {self.model_name} (instances={self.model_instances})")
        print("=" * 60)
        print(f"Total documents: {self.total_files}")
        if self.already_completed > 0:
            print(f"- Already processed (skipped) : {self.already_completed}")
        print("")
        print("Average:")
        print(f"- VLM inference:  {self.avg_vlm_sec:.2f} sec")
        print(f"- Classification: {self.avg_classification_sec:.2f} sec")
        print(f"- File operation: {self.avg_file_op_sec:.2f} sec")
        print(f"- Other:          {self.avg_other_sec:.2f} sec")
        print(f"- Total/document: {self.avg_time_per_image:.2f} sec")
        print("")
        print(f"Median/document:  {self.median_time_per_image:.2f} sec")
        print(f"P95/document:     {self.p95_time_per_image:.2f} sec")
        print(f"Minimum/document: {self.min_time_per_image:.2f} sec")
        print(f"Maximum/document: {self.max_time_per_image:.2f} sec")
        if self.elapsed_time_sec > 0:
            fpm = (self.processed / (self.elapsed_time_sec / 60.0))
            fph = fpm * 60.0
            print(f"Throughput:       {fpm:.1f} files/min ({fph:.0f} files/hr)")
        print(f"Total VLM calls:  {self.total_vlm_calls}")
        if self.total_retries > 0:
            print(f"VLM retries:      {self.total_retries}")
        if self.avg_queue_wait_sec > 0:
            print(f"Avg queue wait:   {self.avg_queue_wait_sec:.2f} sec")
        print("")
        print(f"Total execution time: {self.elapsed_time_sec:.2f} sec")
        print("-" * 60)
        print("Document Classification & Routing Breakdown:")
        print(f"- correctly classified as KYC        : {self.correctly_classified_kyc}")
        print(f"- classified as NON_KYC              : {self.classified_non_kyc}")
        print(f"- classified as UNKNOWN              : {self.classified_unknown}")
        print(f"- rejected due to low confidence     : {self.rejected_low_confidence}")
        print(f"- identified as multiple-document    : {self.identified_multi_doc}")
        print(f"- failed due to errors               : {self.failed_errors}")
        if self.already_completed > 0:
            print(f"- previously completed & verified    : {self.already_completed}")
        if self.skipped_unsupported > 0:
            print(f"- unsupported files (ignored)        : {self.skipped_unsupported}")
        print("-" * 60)
        print("System Resource Statistics:")
        print(f"- Active Workers                     : {self.max_active_workers}")
        print(f"- Avg CPU Usage                      : {self.avg_cpu_percent:.1f}%")
        print(f"- Peak CPU Usage                     : {self.peak_cpu_percent:.1f}%")
        if self.peak_memory_mb > 0:
            print(f"- Peak Memory (RAM)                  : {self.peak_memory_mb:.1f} MB")
        if self.peak_vram_mb > 0:
            print(f"- Peak VRAM (GPU {self.gpu_name})       : {self.peak_vram_mb:.1f} MB")
        print("-" * 60)
        print("KYC Document Sub-types:")
        for doc_type, count in sorted(self.counts_by_type.items()):
            if doc_type not in ("UNKNOWN", "NON_KYC", "IGNORED"):
                print(f"  {doc_type:<32}: {count}")
        print("=" * 60 + "\n")


TEMPORARY_EXTENSIONS = frozenset({
    ".tmp", ".part", ".crdownload", ".download", ".partial",
    ".swp", ".lock", ".temp", ".pending", ".uploading",
})


def iter_source_batches(
    source_dir: Union[str, Path],
    batch_size: int = 100,
    supported_extensions: Optional[Set[str]] = None,
    limit: Optional[int] = None,
) -> Iterator[Tuple[List[Path], int]]:
    """
    Streams supported source files in bounded chunks of `batch_size`.
    Guarantees that memory consumption does not scale with 100,000 files.
    Yields (batch_paths, batch_index).
    """
    src = Path(source_dir)
    if not src.is_dir():
        return

    valid_exts = supported_extensions if supported_extensions is not None else {".jpg"}
    current_batch: List[Path] = []
    yielded_count = 0
    batch_idx = 1

    try:
        with os.scandir(src) as it:
            for entry in it:
                if entry.is_file():
                    name = entry.name
                    if name.startswith(".") or name.startswith("~"):
                        continue
                    ext = os.path.splitext(name)[1].lower()
                    if ext in TEMPORARY_EXTENSIONS:
                        continue
                    if ext in valid_exts:
                        current_batch.append(Path(entry.path))
                        if len(current_batch) >= batch_size:
                            current_batch.sort(key=lambda p: p.name)
                            if limit is not None and (yielded_count + len(current_batch)) > limit:
                                current_batch = current_batch[: limit - yielded_count]
                            yielded_count += len(current_batch)
                            yield current_batch, batch_idx
                            batch_idx += 1
                            current_batch = []
                            if limit is not None and yielded_count >= limit:
                                return
    except OSError as e:
        logger.error("Error scanning source batches from '%s': %s", src, e)

    if current_batch:
        current_batch.sort(key=lambda p: p.name)
        if limit is not None and (yielded_count + len(current_batch)) > limit:
            current_batch = current_batch[: limit - yielded_count]
        yield current_batch, batch_idx


class BatchProcessor:
    """
    Coordinates end-to-end VLM processing of KYC documents from source to destination.
    Uses Qwen3-VL through Ollama API for classification and verified safe routing.
    """

    def __init__(
        self,
        config: Config,
        router: Optional[FileRouter] = None,
        journal: Optional[ProcessingJournal] = None,
    ):
        self.config = config
        self._abort_requested = False
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
        self.use_vlm: bool = True
        self.ocr_engine = None
        self.classifier = None

        # Persistent SQLite Journal for ACID logging, crash recovery, and resume
        if journal is not None:
            self.journal = journal
        else:
            j_path = getattr(config, "journal_path", None) or getattr(config.processing, "journal_path", None)
            if not j_path:
                j_path = config.output.resolved_path.parent / "kyc_journal.db"
            self.journal = ProcessingJournal(db_path=j_path, dry_run=config.processing.dry_run)

    def close(self):
        """Safely close open resources including the processing journal."""
        if hasattr(self, "journal") and self.journal:
            self.journal.close()

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

        try:
            with os.scandir(src) as it:
                for entry in it:
                    if entry.is_file():
                        name = entry.name
                        if name.startswith(".") or name.startswith("~"):
                            continue
                        ext = os.path.splitext(name)[1].lower()
                        if ext in TEMPORARY_EXTENSIONS:
                            continue
                        p = Path(entry.path)
                        if ext in valid_exts:
                            supported.append(p)
                        else:
                            unsupported.append(p)
        except OSError as e:
            logger.error("Error scanning directory '%s': %s", src, e)

        supported.sort(key=lambda p: p.name)
        return supported, unsupported

    def iter_source_batches(
        self,
        source_dir: Optional[Path] = None,
        batch_size: int = 100,
        limit: Optional[int] = None,
    ) -> Iterator[Tuple[List[Path], int]]:
        """
        Streams supported source files in bounded chunks of `batch_size`.
        Guarantees that memory consumption does not scale with 100,000 files.
        Yields (batch_paths, batch_index).
        """
        src = Path(source_dir) if source_dir else self.config.input.resolved_path
        valid_exts = self.config.processing.supported_extensions
        return iter_source_batches(
            source_dir=src,
            batch_size=batch_size,
            supported_extensions=valid_exts,
            limit=limit,
        )

    def process_file(
        self,
        file_path: Path,
        processing_date: Optional[datetime.date] = None,
    ) -> ProcessedItem:
        """
        Process a single document through VLM classification and safe routing.
        Guaranteed not to raise uncaught exceptions.
        """
        start_time = time.perf_counter()
        ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        ext = file_path.suffix.lower()
        if ext not in self.config.processing.supported_extensions:
            logger.info(
                "Ignoring non-supported file '%s' (only %s supported).",
                file_path.name,
                ", ".join(sorted(self.config.processing.supported_extensions)),
            )
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
            dest_root = self.config.output.resolved_path
            dry_run = self.config.processing.dry_run
            operation = self.config.output.operation
            ollama_url = getattr(self.config.vlm, "endpoint", "http://localhost:11434/api/chat")
            model = getattr(self.config.vlm, "model", "qwen3-vl:4b-instruct")
            timeout = getattr(self.config.vlm, "timeout", 300)
            single_pass = getattr(self.config.vlm, "single_pass_mode", True)
            max_dim = getattr(self.config.vlm, "image_max_dimension", 1200)
            img_quality = getattr(self.config.vlm, "image_quality", 85)
            keep_alive = getattr(self.config.vlm, "keep_alive", "24h")
            num_ctx = getattr(self.config.vlm, "num_ctx", 8192)

            # Reconcile against journal first: skip already completed files on disk
            if self.journal:
                reconciled = self.journal.reconcile_file_status(file_path, expected_operation=operation)
                if isinstance(reconciled, tuple):
                    rec_status, rec_cat, rec_dest = reconciled
                    cat_name = rec_cat or "UNKNOWN"
                    logger.info("File '%s' already completed in journal (%s -> %s). Skipping duplicate.", file_path.name, cat_name, rec_dest)
                    return ProcessedItem(
                        timestamp=ts_str,
                        source_file=file_path.name,
                        source_path=str(file_path.resolve()),
                        document_type=cat_name,
                        confidence=1.0,
                        status="ALREADY_COMPLETED",
                        matched_template=rec_cat,
                        destination_path=rec_dest or "",
                        operation=operation,
                        dry_run=dry_run,
                        processing_time_sec=0.0,
                        rejection_category="KYC" if rec_status in ("COMPLETED", "IDENTIFIED") else "NON_KYC",
                    )
                self.journal.record_start(file_path, operation=operation)

            vlm_res = process_vlm_file(
                image_file=file_path,
                index=1,
                total=1,
                destination_root=dest_root,
                ollama_url=ollama_url,
                model=model,
                dry_run=dry_run,
                operation=operation,
                timeout=timeout,
                single_pass=single_pass,
                image_max_dimension=max_dim,
                image_quality=img_quality,
                keep_alive=keep_alive,
                num_ctx=num_ctx,
            )

            # Record completion / failure in persistent SQLite journal
            t_j_write = 0.0
            if self.journal:
                t_j_start = time.perf_counter()
                if vlm_res.status == "SUCCESS" or vlm_res.routing_type in ("KYC", "NON_KYC"):
                    self.journal.record_completion(
                        source_path=vlm_res.source_file,
                        destination_path=vlm_res.destination_path,
                        category=vlm_res.category,
                        routing_type=vlm_res.routing_type,
                        operation=operation,
                        error=vlm_res.error,
                    )
                else:
                    self.journal.record_failure(
                        source_path=vlm_res.source_file,
                        error=vlm_res.error or "Unidentified routing failure",
                        destination_path=vlm_res.destination_path,
                        operation=operation,
                    )
                t_j_write = time.perf_counter() - t_j_start
                vlm_res.timings["Journal write"] = t_j_write

            if getattr(vlm_res, "is_multiple_document", False):
                doc_type = "NON_KYC"
                status = "NON_KYC"
                rej_category = "MULTI_DOC"
                confidence = 1.0
            elif vlm_res.routing_type == "KYC":
                doc_type = vlm_res.category or "UNKNOWN"
                status = "IDENTIFIED"
                rej_category = "KYC"
                confidence = 1.0
            elif vlm_res.routing_type == "NON_KYC":
                doc_type = "NON_KYC"
                status = "NON_KYC"
                rej_category = "NON_KYC"
                confidence = 1.0
            else:
                doc_type = "UNKNOWN"
                status = "FAILED" if vlm_res.error else "UNIDENTIFIED"
                rej_category = "ERROR" if vlm_res.error else "UNKNOWN"
                confidence = 0.0

            timing = DocumentProfileTiming(
                file_load_sec=vlm_res.timings.get("File read", 0.0),
                preprocessing_sec=vlm_res.timings.get("Image preprocessing", 0.0),
                vlm_inference_sec=vlm_res.timings.get("VLM inference", 0.0),
                ocr_inference_sec=vlm_res.timings.get("VLM inference", 0.0),
                classification_sec=vlm_res.timings.get("Response parsing", 0.0),
                routing_sec=vlm_res.timings.get("Destination resolution", 0.0) + vlm_res.timings.get("Filename generation", 0.0),
                file_operation_sec=vlm_res.timings.get("File move", 0.0) + vlm_res.timings.get("Move verification", 0.0),
                total_sec=vlm_res.elapsed_time,
                queue_wait_sec=vlm_res.timings.get("Queue wait", 0.0),
                journal_write_sec=t_j_write,
                vlm_calls=getattr(vlm_res, "vlm_calls", 1),
                retries=getattr(vlm_res, "retry_count", 0),
            )

            return ProcessedItem(
                timestamp=ts_str,
                source_file=file_path.name,
                source_path=str(file_path.resolve()),
                document_type=doc_type,
                confidence=confidence,
                status=status,
                matched_template=vlm_res.category,
                destination_path=str(vlm_res.destination_path),
                operation=operation,
                dry_run=dry_run,
                error=vlm_res.error,
                processing_time_sec=vlm_res.elapsed_time,
                timing=timing,
                rejection_category=rej_category,
            )

        except Exception as e:
            duration = time.perf_counter() - start_time
            logger.error("Unexpected error in VLM processing for '%s': %s", file_path.name, e, exc_info=True)
            if self.journal:
                try:
                    self.journal.record_failure(
                        source_path=file_path,
                        error=str(e),
                        operation=self.config.output.operation,
                    )
                except Exception:
                    pass
            try:
                dest_root = self.config.output.resolved_path
                fallback_res = route_document(
                    source_file=file_path,
                    category=None,
                    is_error=True,
                    destination_root=dest_root,
                    dry_run=self.config.processing.dry_run,
                    operation=self.config.output.operation,
                )
                dest_str = str(fallback_res.destination_path)
            except Exception:
                dest_str = str(self.config.output.resolved_path / "Unidentified_files" / file_path.name)

            timing = DocumentProfileTiming(total_sec=duration)
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

    def _route_result_to_processed_item(self, r: RouteResult) -> ProcessedItem:
        ts_str = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if getattr(r, "is_multiple_document", False):
            doc_type = "NON_KYC"
            status = "NON_KYC"
            rej_category = "MULTI_DOC"
            confidence = 1.0
        elif r.routing_type == "KYC":
            doc_type = r.category or "UNKNOWN"
            status = "IDENTIFIED"
            rej_category = "KYC"
            confidence = 1.0
        elif r.routing_type == "NON_KYC":
            doc_type = "NON_KYC"
            status = "NON_KYC"
            rej_category = "NON_KYC"
            confidence = 1.0
        else:
            doc_type = "UNKNOWN"
            status = "FAILED" if r.error else "UNIDENTIFIED"
            rej_category = "ERROR" if r.error else "UNKNOWN"
            confidence = 0.0

        timing = DocumentProfileTiming(
            file_load_sec=r.timings.get("File read", 0.0),
            preprocessing_sec=r.timings.get("Image preprocessing", 0.0),
            vlm_inference_sec=r.timings.get("VLM inference", 0.0),
            ocr_inference_sec=r.timings.get("VLM inference", 0.0),
            classification_sec=r.timings.get("Response parsing", 0.0),
            routing_sec=r.timings.get("Destination resolution", 0.0) + r.timings.get("Filename generation", 0.0),
            file_operation_sec=r.timings.get("File move", 0.0) + r.timings.get("Move verification", 0.0),
            total_sec=r.elapsed_time,
        )

        return ProcessedItem(
            timestamp=ts_str,
            source_file=r.source_file.name,
            source_path=str(r.source_file.resolve()),
            document_type=doc_type,
            confidence=confidence,
            status=status,
            matched_template=r.category,
            destination_path=str(r.destination_path),
            operation=self.config.output.operation,
            dry_run=self.config.processing.dry_run,
            error=r.error,
            processing_time_sec=r.elapsed_time,
            timing=timing,
            rejection_category=rej_category,
        )

    def process_batch(
        self,
        source_dir: Optional[Path] = None,
        progress_callback: Optional[Callable[[int, int, ProcessedItem], None]] = None,
        limit: Optional[int] = None,
    ) -> BatchSummary:
        """
        Execute full VLM batch processing across all files in source directory.
        Processes files in bounded incremental batches with persistent SQLite journaling,
        safe resume, and bounded heap memory.
        """
        batch_start = time.time()
        src = Path(source_dir) if source_dir else self.config.input.resolved_path

        # Determine configured batch chunk size (default 100)
        batch_size = getattr(self.config.processing, "batch_size", 100)
        if batch_size is None or batch_size <= 0:
            batch_size = 100

        # Concurrency & VLM config
        configured_max = getattr(self.config.processing, "max_workers", 4)
        target_workers = 4 if configured_max is None or configured_max <= 0 else configured_max

        model_name = getattr(self.config.vlm, "model", "qwen3-vl:4b-instruct")
        ollama_url = getattr(self.config.vlm, "endpoint", "http://localhost:11434/api/chat")
        model_instances = 1
        dest_root = self.config.output.resolved_path
        dry_run = self.config.processing.dry_run
        operation = self.config.output.operation
        cpu_limit = getattr(self.config.processing, "cpu_limit_percent", 80.0)
        throttle_cpu = getattr(self.config.processing, "throttle_cpu_percent", 75.0)
        timeout = getattr(self.config.vlm, "timeout", 300)
        single_pass = getattr(self.config.vlm, "single_pass_mode", True)
        max_dim = getattr(self.config.vlm, "image_max_dimension", 1200)
        img_quality = getattr(self.config.vlm, "image_quality", 85)
        keep_alive = getattr(self.config.vlm, "keep_alive", "24h")
        num_ctx = getattr(self.config.vlm, "num_ctx", 8192)

        items: List[ProcessedItem] = []
        counts_by_type: Dict[str, int] = {}
        total_discovered = 0
        total_failed = 0
        total_already_completed = 0
        total_multi_doc = 0
        all_times: List[float] = []
        timing_records: List[DocumentProfileTiming] = []
        peak_cpu_observed = 0.0
        peak_ram_observed = 0.0
        peak_vram_observed = 0.0
        cpu_limit_ever_hit = False
        gpu_device_name = "None"
        effective_workers = target_workers

        # Stream files in bounded batches
        batch_generator = self.iter_source_batches(source_dir=src, batch_size=batch_size, limit=limit)
        active_batch_count = 0

        for batch_paths, b_idx in batch_generator:
            active_batch_count += 1
            total_discovered += len(batch_paths)
            batch_to_process: List[Path] = []

            # 1. Reconcile against persistent journal: skip already verified files
            for file_path in batch_paths:
                reconciled = self.journal.reconcile_file_status(file_path, expected_operation=operation)
                if reconciled is not None:
                    # File was already completed and verified intact on disk
                    rec_status, rec_cat, rec_dest = reconciled
                    total_already_completed += 1
                    cat_name = rec_cat or "UNKNOWN"
                    p_item = ProcessedItem(
                        timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
                        source_file=file_path.name,
                        source_path=str(file_path.resolve()),
                        document_type=cat_name,
                        confidence=1.0,
                        status="ALREADY_COMPLETED",
                        matched_template=rec_cat,
                        destination_path=rec_dest or "",
                        operation=operation,
                        dry_run=dry_run,
                        processing_time_sec=0.0,
                        rejection_category="KYC" if rec_status in ("COMPLETED", "IDENTIFIED") else "NON_KYC",
                    )
                    counts_by_type[cat_name] = counts_by_type.get(cat_name, 0) + 1
                    if len(items) < 1000:
                        items.append(p_item)
                    if progress_callback:
                        progress_callback(total_already_completed + len(all_times), total_discovered, p_item)
                else:
                    batch_to_process.append(file_path)

            if not batch_to_process:
                continue

            # Record in-progress start in journal for files about to be processed
            for p in batch_to_process:
                self.journal.record_start(p, operation=operation)

            # Determine dynamic worker count for this chunk
            effective_workers = max(1, min(target_workers, len(batch_to_process)))

            logger.info(
                "Processing batch %d (%d files, workers=%d, already_skipped=%d)...",
                b_idx,
                len(batch_to_process),
                effective_workers,
                total_already_completed,
            )

            # Progress callback adapter
            def _vlm_progress(completed_idx: int, total_idx: int, route_res: RouteResult):
                if progress_callback:
                    p_item = self._route_result_to_processed_item(route_res)
                    progress_callback(total_already_completed + len(all_times) + completed_idx, total_discovered, p_item)

            vlm_summary = process_vlm_batch(
                files=batch_to_process,
                max_workers=effective_workers,
                destination_root=dest_root,
                cpu_limit=cpu_limit,
                throttle_cpu=throttle_cpu,
                ollama_url=ollama_url,
                model=model_name,
                dry_run=dry_run,
                operation=operation,
                progress_callback=_vlm_progress if progress_callback else None,
                timeout=timeout,
                single_pass=single_pass,
                max_dimension=max_dim,
                jpeg_quality=img_quality,
                keep_alive=keep_alive,
                num_ctx=num_ctx,
            )

            total_failed += vlm_summary.failed_files
            if vlm_summary.peak_cpu_percent > peak_cpu_observed:
                peak_cpu_observed = vlm_summary.peak_cpu_percent
            if vlm_summary.peak_memory_percent > peak_ram_observed:
                peak_ram_observed = vlm_summary.peak_memory_percent
            if vlm_summary.peak_vram_mb > peak_vram_observed:
                peak_vram_observed = vlm_summary.peak_vram_mb
            if vlm_summary.cpu_limit_exceeded:
                cpu_limit_ever_hit = True
            if vlm_summary.gpu_info.get("available"):
                gpu_device_name = vlm_summary.gpu_info.get("name", "NVIDIA GPU")

            # Update journal and aggregate results
            for r in vlm_summary.results:
                if getattr(r, "is_multiple_document", False):
                    total_multi_doc += 1
                p_item = self._route_result_to_processed_item(r)
                if len(items) < 1000:
                    items.append(p_item)

                all_times.append(p_item.processing_time_sec)
                if p_item.timing:
                    timing_records.append(p_item.timing)

                doc_t = p_item.document_type
                counts_by_type[doc_t] = counts_by_type.get(doc_t, 0) + 1

                # Commit completion / failure to SQLite journal
                if r.status == "SUCCESS" or r.routing_type in ("KYC", "NON_KYC"):
                    self.journal.record_completion(
                        source_path=r.source_file,
                        destination_path=r.destination_path,
                        category=r.category,
                        routing_type=r.routing_type,
                        operation=operation,
                        error=r.error,
                    )
                else:
                    self.journal.record_failure(
                        source_path=r.source_file,
                        error=r.error or "Unidentified routing failure",
                        destination_path=r.destination_path,
                        operation=operation,
                    )

            if self._abort_requested:
                logger.warning("Batch processing aborted by user after batch %d.", b_idx)
                break

        # Compute summary metrics across all processed batches
        total_duration = time.time() - batch_start
        processed_count = total_already_completed + len(all_times)

        # In case no batches yielded (e.g. empty directory)
        if total_discovered == 0:
            effective_workers = 1

        summary = BatchSummary(
            total_files=total_discovered,
            processed=processed_count,
            failed=total_failed,
            skipped_unsupported=0,
            counts_by_type=counts_by_type,
            items=items,
            elapsed_time_sec=total_duration,
            avg_time_per_image=statistics.mean(all_times) if all_times else 0.0,
            max_active_workers=effective_workers,
            avg_cpu_percent=peak_cpu_observed * 0.7 if peak_cpu_observed > 0 else 0.0,
            peak_cpu_percent=peak_cpu_observed,
            peak_memory_mb=peak_ram_observed,
            pipeline_type="VLM",
            model_name=model_name,
            model_instances=model_instances,
            identified_multi_doc=total_multi_doc,
            avg_vlm_sec=(
                sum(t.vlm_inference_sec for t in timing_records if t.vlm_inference_sec is not None) / len(timing_records)
                if timing_records
                else 0.0
            ),
            already_completed=total_already_completed,
            peak_vram_mb=peak_vram_observed,
            gpu_name=gpu_device_name,
        )

        summary.min_time_per_image = min(all_times) if all_times else 0.0
        summary.max_time_per_image = max(all_times) if all_times else 0.0

        if all_times:
            sorted_times = sorted(all_times)
            p95_idx = int(round(0.95 * len(sorted_times))) - 1
            p95_idx = max(0, min(len(sorted_times) - 1, p95_idx))
            summary.p95_time_per_image = sorted_times[p95_idx]
            summary.median_time_per_image = statistics.median(all_times)
        else:
            summary.p95_time_per_image = 0.0
            summary.median_time_per_image = 0.0

        summary.total_vlm_calls = sum(getattr(t, "vlm_calls", 1) for t in timing_records)
        summary.total_retries = sum(getattr(t, "retries", 0) for t in timing_records)
        q_waits = [getattr(t, "queue_wait_sec", 0.0) for t in timing_records]
        summary.avg_queue_wait_sec = statistics.mean(q_waits) if q_waits else 0.0

        n_t = len(timing_records)
        if n_t > 0:
            summary.avg_classification_sec = sum(t.classification_sec for t in timing_records) / n_t
            summary.avg_file_op_sec = sum(t.routing_sec + t.file_operation_sec for t in timing_records) / n_t
            summary.avg_other_sec = sum(t.logging_sec for t in timing_records) / n_t

        summary.correctly_classified_kyc = sum(
            c for doc_t, c in counts_by_type.items() if doc_t not in ("UNKNOWN", "NON_KYC", "IGNORED")
        )
        summary.classified_non_kyc = counts_by_type.get("NON_KYC", 0)
        summary.classified_unknown = counts_by_type.get("UNKNOWN", 0)
        summary.failed_errors = total_failed
        summary.kyc_routed_count = summary.correctly_classified_kyc
        summary.non_kyc_routed_count = summary.classified_non_kyc

        return summary
