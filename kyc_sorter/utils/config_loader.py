"""
Configuration loader and validator for the KYC Document Sorter.
Provides typed access, validation, path resolution, and runtime overrides.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import yaml


@dataclass
class ApplicationConfig:
    name: str = "KYC Document Sorter"
    version: str = "1.0.0"


@dataclass
class InputConfig:
    directory: str = "Source Files"
    resolved_path: Path = field(init=False)

    def resolve(self, base_dir: Path) -> Path:
        p = Path(self.directory)
        if p.is_absolute():
            self.resolved_path = p
            return p
        for candidate_base in (base_dir, Path.cwd(), base_dir.parent):
            candidate = (candidate_base / p).resolve()
            if candidate.exists():
                self.resolved_path = candidate
                return candidate
        self.resolved_path = (base_dir / p).resolve()
        return self.resolved_path


@dataclass
class OutputConfig:
    directory: str = "Destination files"
    operation: str = "copy"  # "copy" or "move"
    create_date_folder: bool = False
    date_format: str = "%Y-%m-%d"
    kyc_folder: str = "KYC"
    unknown_folder: str = "NON KYC"
    generate_companion_files: bool = True
    resolved_path: Path = field(init=False)

    def resolve(self, base_dir: Path) -> Path:
        p = Path(self.directory)
        if p.is_absolute():
            self.resolved_path = p
            return p
        for candidate_base in (base_dir, Path.cwd(), base_dir.parent):
            candidate = (candidate_base / p).resolve()
            if candidate.exists():
                self.resolved_path = candidate
                return candidate
        self.resolved_path = (base_dir / p).resolve()
        return self.resolved_path


@dataclass
class ProcessingConfig:
    confidence_threshold: float = 0.80
    dry_run: bool = False
    max_workers: int = 6
    cpu_limit_percent: float = 80.0
    adaptive_workers: bool = True
    multiple_document_score_margin: float = 0.10
    limit: Optional[int] = None
    supported_extensions: Set[str] = field(
        default_factory=lambda: {".jpg"}
    )


@dataclass
class WatcherConfig:
    stability_delay_sec: float = 1.0
    settle_timeout_sec: float = 30.0
    poll_interval_sec: float = 0.5


@dataclass
class DocumentTypeMetadata:
    folder: str
    filename_suffix: str


DEFAULT_DOCUMENT_METADATA: Dict[str, DocumentTypeMetadata] = {
    "AADHAAR_CARD": DocumentTypeMetadata(folder="AADHAAR CARD", filename_suffix="AADHAAR_CARD"),
    "AADHAAR CARD": DocumentTypeMetadata(folder="AADHAAR CARD", filename_suffix="AADHAAR_CARD"),
    "PAN_CARD": DocumentTypeMetadata(folder="PAN CARD", filename_suffix="PAN_CARD"),
    "PAN CARD": DocumentTypeMetadata(folder="PAN CARD", filename_suffix="PAN_CARD"),
    "VOTER_ID": DocumentTypeMetadata(folder="VOTER ID", filename_suffix="VOTER_ID"),
    "VOTER ID": DocumentTypeMetadata(folder="VOTER ID", filename_suffix="VOTER_ID"),
    "BANK_STATEMENT": DocumentTypeMetadata(folder="BANK STATEMENT", filename_suffix="BANK_STATEMENT"),
    "BANK STATEMENT": DocumentTypeMetadata(folder="BANK STATEMENT", filename_suffix="BANK_STATEMENT"),
    "BANK_PASSBOOK": DocumentTypeMetadata(folder="BANK STATEMENT", filename_suffix="BANK_STATEMENT"),
    "CKYC_CONFIRMATION": DocumentTypeMetadata(folder="CKYC CONFIRMATION", filename_suffix="CKYC_CONFIRMATION"),
    "CKYC CONFIRMATION": DocumentTypeMetadata(folder="CKYC CONFIRMATION", filename_suffix="CKYC_CONFIRMATION"),
    "DRIVING_LICENCE": DocumentTypeMetadata(folder="DRIVING LICENCE", filename_suffix="DRIVING_LICENCE"),
    "DRIVING LICENCE": DocumentTypeMetadata(folder="DRIVING LICENCE", filename_suffix="DRIVING_LICENCE"),
    "ELECTRICITY_BILL": DocumentTypeMetadata(folder="ELECTRICITY BILL", filename_suffix="ELECTRICITY_BILL"),
    "ELECTRICITY BILL": DocumentTypeMetadata(folder="ELECTRICITY BILL", filename_suffix="ELECTRICITY_BILL"),
    "GAS_BILL": DocumentTypeMetadata(folder="GAS BILL", filename_suffix="GAS_BILL"),
    "GAS BILL": DocumentTypeMetadata(folder="GAS BILL", filename_suffix="GAS_BILL"),
    "MUNICIPAL_TAX_RECEIPT": DocumentTypeMetadata(folder="MUNICIPAL TAX RECEIPT", filename_suffix="MUNICIPAL_RECEIPT"),
    "MUNICIPAL TAX RECEIPT": DocumentTypeMetadata(folder="MUNICIPAL TAX RECEIPT", filename_suffix="MUNICIPAL_RECEIPT"),
    "MUNICIPAL_RECEIPT": DocumentTypeMetadata(folder="MUNICIPAL TAX RECEIPT", filename_suffix="MUNICIPAL_RECEIPT"),
    "PHONE_BILL": DocumentTypeMetadata(folder="PHONE BILL", filename_suffix="PHONE_BILL"),
    "PHONE BILL": DocumentTypeMetadata(folder="PHONE BILL", filename_suffix="PHONE_BILL"),
    "RATION_CARD": DocumentTypeMetadata(folder="RATION CARD", filename_suffix="RATION_CARD"),
    "RATION CARD": DocumentTypeMetadata(folder="RATION CARD", filename_suffix="RATION_CARD"),
    "REGISTRATION_CERTIFICATE": DocumentTypeMetadata(folder="REGISTRATION CERTIFICATE", filename_suffix="REGISTRATION_CERTIFICATE"),
    "REGISTRATION CERTIFICATE": DocumentTypeMetadata(folder="REGISTRATION CERTIFICATE", filename_suffix="REGISTRATION_CERTIFICATE"),
    "PASSPORT": DocumentTypeMetadata(folder="PASSPORT", filename_suffix="PASSPORT"),
    "GOVT_ID": DocumentTypeMetadata(folder="GOVT ID", filename_suffix="GOVT_ID"),
    "GOVT ID": DocumentTypeMetadata(folder="GOVT ID", filename_suffix="GOVT_ID"),
    "CKYC_DECLARATION": DocumentTypeMetadata(folder="CKYC DECLARATION", filename_suffix="CKYC_DECLARATION"),
    "CKYC DECLARATION": DocumentTypeMetadata(folder="CKYC DECLARATION", filename_suffix="CKYC_DECLARATION"),
    "IT_RETURNS": DocumentTypeMetadata(folder="IT RETURNS", filename_suffix="IT_RETURNS"),
    "IT RETURNS": DocumentTypeMetadata(folder="IT RETURNS", filename_suffix="IT_RETURNS"),
    "UNKNOWN": DocumentTypeMetadata(folder="NON KYC", filename_suffix="UNKNOWN"),
}

@dataclass
class OCRConfig:
    language: str = "en"
    use_gpu: bool = False
    cpu_threads: int = 3
    text_detection_model_name: str = "PP-OCRv5_mobile_det"
    text_recognition_model_name: str = "en_PP-OCRv5_mobile_rec"
    enable_mkldnn: bool = False
    use_angle_classification: bool = False


@dataclass
class SecurityConfig:
    log_sensitive_text: bool = False
    store_raw_ocr_debug: bool = False


@dataclass
class Config:
    application: ApplicationConfig = field(default_factory=ApplicationConfig)
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    ocr: OCRConfig = field(default_factory=OCRConfig)
    security: SecurityConfig = field(default_factory=SecurityConfig)
    kyc_folder: str = "KYC"
    unknown_folder: str = "NON KYC"
    unknown_suffix: str = "UNKNOWN"
    documents: Dict[str, DocumentTypeMetadata] = field(default_factory=lambda: dict(DEFAULT_DOCUMENT_METADATA))
    folders: Dict[str, str] = field(
        default_factory=lambda: {k: v.folder for k, v in DEFAULT_DOCUMENT_METADATA.items()}
    )
    suffixes: Dict[str, str] = field(
        default_factory=lambda: {k: v.filename_suffix for k, v in DEFAULT_DOCUMENT_METADATA.items()}
    )
    base_dir: Path = field(default_factory=lambda: Path.cwd())

    def get_destination_subfolder(self, doc_type: str) -> str:
        """Return configured folder name for the given doc_type, defaulting to unknown folder."""
        normalized = (doc_type or "").strip().upper()
        if normalized in self.documents:
            return self.documents[normalized].folder
        return self.folders.get(normalized, self.unknown_folder)

    def get_filename_suffix(self, doc_type: str) -> str:
        """Return configured filename suffix for the given doc_type, defaulting to unknown suffix."""
        normalized = (doc_type or "").strip().upper()
        if normalized in self.documents:
            return self.documents[normalized].filename_suffix
        return self.suffixes.get(normalized, self.unknown_suffix)


def _deep_update(mapping: Dict[str, Any], updating_mapping: Dict[str, Any]) -> Dict[str, Any]:
    """Recursively update a dict."""
    for k, v in updating_mapping.items():
        if isinstance(v, dict) and k in mapping and isinstance(mapping[k], dict):
            _deep_update(mapping[k], v)
        else:
            mapping[k] = v
    return mapping


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    base_dir: Optional[Union[str, Path]] = None,
) -> Config:
    """
    Load configuration from YAML file with environment variable fallback and validation.
    
    Args:
        config_path: Path to config.yaml. If None, looks in standard locations.
        overrides: Dictionary of values to override loaded config.
        base_dir: Base directory for resolving relative paths. Defaults to config file parent or cwd.
    
    Returns:
        Config: Populated and validated configuration object.
    """
    raw_data: Dict[str, Any] = {}
    actual_path: Optional[Path] = None

    if config_path:
        p = Path(config_path)
        if p.is_file():
            actual_path = p.resolve()
    else:
        # Check standard locations
        candidates = [
            Path("config.yaml"),
            Path(__file__).resolve().parent.parent / "config.yaml",
            Path.cwd() / "config.yaml",
        ]
        for c in candidates:
            if c.is_file():
                actual_path = c.resolve()
                break

    if actual_path and actual_path.is_file():
        with open(actual_path, "r", encoding="utf-8") as f:
            loaded = yaml.safe_load(f)
            if isinstance(loaded, dict):
                raw_data = loaded

    if overrides:
        _deep_update(raw_data, overrides)

    # Determine base directory for relative path resolution
    if base_dir:
        resolved_base = Path(base_dir).resolve()
    elif actual_path:
        resolved_base = actual_path.parent.resolve()
    else:
        resolved_base = Path.cwd().resolve()

    # Application section
    app_data = raw_data.get("application", {})
    app_cfg = ApplicationConfig(
        name=str(app_data.get("name", "KYC Document Sorter")),
        version=str(app_data.get("version", "1.0.0")),
    )

    # Input section
    in_data = raw_data.get("input", {})
    in_cfg = InputConfig(directory=str(in_data.get("directory", "Source Files")))
    in_cfg.resolve(resolved_base)

    # Output section
    out_data = raw_data.get("output", {})
    op = str(out_data.get("operation", "copy")).strip().lower()
    if op not in ("copy", "move"):
        op = "copy"  # Default to copy if invalid for safety
    kyc_f = str(out_data.get("kyc_folder", "KYC")).strip()
    out_cfg = OutputConfig(
        directory=str(out_data.get("directory", "Destination files")),
        operation=op,
        create_date_folder=bool(out_data.get("create_date_folder", False)),
        date_format=str(out_data.get("date_format", "%Y-%m-%d")),
        kyc_folder=kyc_f,
        unknown_folder=str(out_data.get("unknown_folder", "NON KYC")).strip(),
        generate_companion_files=bool(out_data.get("generate_companion_files", True)),
    )
    out_cfg.resolve(resolved_base)

    # Processing section
    proc_data = raw_data.get("processing", {})
    class_data = raw_data.get("classification", {})
    try:
        conf_thresh = float(proc_data.get("confidence_threshold", 0.80))
        conf_thresh = max(0.0, min(1.0, conf_thresh))
    except (TypeError, ValueError):
        conf_thresh = 0.80

    raw_exts = proc_data.get("supported_extensions", [".jpg"])
    clean_exts: Set[str] = set()
    for ext in raw_exts:
        ext_str = str(ext).strip().lower()
        if not ext_str.startswith("."):
            ext_str = f".{ext_str}"
        clean_exts.add(ext_str)

    try:
        max_workers = int(proc_data.get("max_workers", 6))
        max_workers = max(1, max_workers)
    except (TypeError, ValueError):
        max_workers = 6

    try:
        cpu_limit = float(proc_data.get("cpu_limit_percent", 80.0))
        cpu_limit = max(10.0, min(100.0, cpu_limit))
    except (TypeError, ValueError):
        cpu_limit = 80.0

    adaptive_workers = bool(proc_data.get("adaptive_workers", True))

    score_margin_raw = proc_data.get(
        "multiple_document_score_margin",
        class_data.get("multiple_document_score_margin", 0.10),
    )
    try:
        score_margin = float(score_margin_raw)
        score_margin = max(0.0, min(1.0, score_margin))
    except (TypeError, ValueError):
        score_margin = 0.10

    limit_raw = proc_data.get("limit")
    limit_val: Optional[int] = None
    if limit_raw is not None:
        try:
            limit_val = int(limit_raw)
            if limit_val <= 0:
                limit_val = None
        except (TypeError, ValueError):
            limit_val = None

    proc_cfg = ProcessingConfig(
        confidence_threshold=conf_thresh,
        dry_run=bool(proc_data.get("dry_run", False)),
        max_workers=max_workers,
        cpu_limit_percent=cpu_limit,
        adaptive_workers=adaptive_workers,
        multiple_document_score_margin=score_margin,
        limit=limit_val,
        supported_extensions=clean_exts,
    )

    # Watcher section
    watch_data = raw_data.get("watcher", {})
    watcher_cfg = WatcherConfig(
        stability_delay_sec=float(watch_data.get("stability_delay_sec", 1.0)),
        settle_timeout_sec=float(watch_data.get("settle_timeout_sec", 30.0)),
        poll_interval_sec=float(watch_data.get("poll_interval_sec", 0.5)),
    )

    # OCR section
    ocr_data = raw_data.get("ocr", {})
    ocr_cfg = OCRConfig(
        language=str(ocr_data.get("language", "en")),
        use_gpu=bool(ocr_data.get("use_gpu", ocr_data.get("use_cuda", False))),
        cpu_threads=int(ocr_data.get("cpu_threads", 3)),
        text_detection_model_name=str(ocr_data.get("text_detection_model_name", "PP-OCRv5_mobile_det")),
        text_recognition_model_name=str(ocr_data.get("text_recognition_model_name", "en_PP-OCRv5_mobile_rec")),
        enable_mkldnn=bool(ocr_data.get("enable_mkldnn", False)),
        use_angle_classification=bool(ocr_data.get("use_angle_classification", False)),
    )

    # Security section
    sec_data = raw_data.get("security", {})
    sec_cfg = SecurityConfig(
        log_sensitive_text=bool(sec_data.get("log_sensitive_text", False)),
        store_raw_ocr_debug=bool(sec_data.get("store_raw_ocr_debug", False)),
    )

    # Unknown / NON KYC settings
    unk_data = raw_data.get("unknown", {})
    unk_folder = str(unk_data.get("folder", out_data.get("unknown_folder", "NON KYC"))).strip()
    unk_suffix = str(unk_data.get("filename_suffix", "UNKNOWN")).strip()

    # Document type mappings (folders and filename suffixes)
    docs_map: Dict[str, DocumentTypeMetadata] = dict(DEFAULT_DOCUMENT_METADATA)
    folders_map: Dict[str, str] = {k: v.folder for k, v in DEFAULT_DOCUMENT_METADATA.items()}
    suffixes_map: Dict[str, str] = {k: v.filename_suffix for k, v in DEFAULT_DOCUMENT_METADATA.items()}

    # Merge custom 'documents' dict if specified
    custom_docs = raw_data.get("documents", {})
    if isinstance(custom_docs, dict):
        for raw_k, meta in custom_docs.items():
            k = str(raw_k).strip().upper()
            if isinstance(meta, dict):
                f_name = str(meta.get("folder", folders_map.get(k, k)))
                s_name = str(meta.get("filename_suffix", suffixes_map.get(k, k)))
                docs_map[k] = DocumentTypeMetadata(folder=f_name, filename_suffix=s_name)
                folders_map[k] = f_name
                suffixes_map[k] = s_name

    # Merge legacy 'folders' dict if specified
    custom_folders = raw_data.get("folders", {})
    if isinstance(custom_folders, dict):
        for raw_k, f_val in custom_folders.items():
            k = str(raw_k).strip().upper()
            f_str = str(f_val).strip()
            folders_map[k] = f_str
            if k in docs_map:
                docs_map[k].folder = f_str
            else:
                docs_map[k] = DocumentTypeMetadata(folder=f_str, filename_suffix=suffixes_map.get(k, f_str))

    # Ensure UNKNOWN is present
    if "UNKNOWN" not in folders_map:
        folders_map["UNKNOWN"] = unk_folder
    if "UNKNOWN" not in suffixes_map:
        suffixes_map["UNKNOWN"] = unk_suffix
    if "UNKNOWN" not in docs_map:
        docs_map["UNKNOWN"] = DocumentTypeMetadata(folder=unk_folder, filename_suffix=unk_suffix)

    return Config(
        application=app_cfg,
        input=in_cfg,
        output=out_cfg,
        processing=proc_cfg,
        watcher=watcher_cfg,
        ocr=ocr_cfg,
        security=sec_cfg,
        kyc_folder=kyc_f,
        unknown_folder=unk_folder,
        unknown_suffix=unk_suffix,
        documents=docs_map,
        folders=folders_map,
        suffixes=suffixes_map,
        base_dir=resolved_base,
    )
