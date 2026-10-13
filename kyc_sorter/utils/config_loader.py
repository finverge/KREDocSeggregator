"""
Configuration loader, validator, and environment resolver for KYC Document Sorter.
Provides typed access, strict validation, safe path resolution, environment variable
support, and precedence handling for local development and containerized server deployments.
"""

from __future__ import annotations

import logging
import os
import urllib.parse
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Union
import yaml

logger = logging.getLogger(__name__)


class ConfigurationError(ValueError):
    """Raised when configuration validation or path resolution fails."""
    pass


def normalize_ollama_endpoint(raw_url: str) -> str:
    """
    Normalizes an Ollama endpoint URL, supporting both base URLs and explicit /api/chat paths.

    Examples:
      - 'http://localhost:11434' -> 'http://localhost:11434/api/chat'
      - 'http://localhost:11434/' -> 'http://localhost:11434/api/chat'
      - 'http://localhost:11434/api/chat' -> 'http://localhost:11434/api/chat'
      - 'http://ollama-server:11434' -> 'http://ollama-server:11434/api/chat'
      - 'localhost:11434' -> 'http://localhost:11434/api/chat'
      - 'https://remote.server:11434/api/chat/' -> 'https://remote.server:11434/api/chat'
    """
    if not raw_url or not str(raw_url).strip():
        raise ConfigurationError("Ollama endpoint URL cannot be empty.")

    url = str(raw_url).strip()
    if "://" in url:
        scheme = url.split("://", 1)[0].lower()
        if scheme not in ("http", "https"):
            raise ConfigurationError(
                f"Invalid scheme '{scheme}' in Ollama endpoint '{raw_url}'. Only 'http://' or 'https://' are supported."
            )
    else:
        url = f"http://{url}"

    parsed = urllib.parse.urlparse(url)
    if parsed.scheme not in ("http", "https"):
        raise ConfigurationError(
            f"Invalid scheme in Ollama endpoint '{raw_url}'. Only 'http://' or 'https://' are supported."
        )
    if not parsed.netloc:
        raise ConfigurationError(
            f"Malformed Ollama endpoint '{raw_url}': missing valid hostname or port."
        )

    path = parsed.path.rstrip("/")
    if not path or path == "":
        path = "/api/chat"
    elif path.endswith("/api"):
        path = f"{path}/chat"
    elif not path.endswith("/api/chat"):
        path = f"{path}/api/chat"

    return urllib.parse.urlunparse((parsed.scheme, parsed.netloc, path, "", "", ""))


def _check_forbidden_paths(p: Path, role: str, allow_legacy_paths: bool = False) -> Path:
    """
    Prevents the application from accessing or mutating the original production project
    (D:/KYC_Document_sorter) or the OCR_VLM backup.

    Fails early with ConfigurationError unless an explicit safe override is authorized.
    """
    p_resolved = p.resolve()
    orig_project = Path("D:/KYC_Document_sorter").resolve()
    ocr_vlm_project = Path("D:/KYC_Document_sorter_OCR_VLM").resolve()
    vlm_only_project = Path(__file__).resolve().parent.parent.parent.resolve()

    is_orig = (
        (p_resolved == orig_project or orig_project in p_resolved.parents)
        and (p_resolved != vlm_only_project and vlm_only_project not in p_resolved.parents)
    )
    is_backup = (
        (p_resolved == ocr_vlm_project or ocr_vlm_project in p_resolved.parents)
        and (p_resolved != vlm_only_project and vlm_only_project not in p_resolved.parents)
    )

    if is_orig or is_backup:
        target_name = "original project" if is_orig else "OCR+VLM backup"
        if not allow_legacy_paths:
            raise ConfigurationError(
                f"Configured {role} directory '{p}' points to forbidden {target_name} ('{p_resolved}'). "
                f"Direct access to legacy project directories is blocked to protect production files. "
                f"Please specify a path within this project or an external mount."
            )
        else:
            rel = p_resolved.relative_to(orig_project if is_orig else ocr_vlm_project)
            redirected = vlm_only_project / rel
            logger.warning(
                "SAFETY INTERVENTION: Config %s path '%s' points to %s. "
                "Redirecting to VLM_ONLY directory '%s'.",
                role, p_resolved, target_name, redirected
            )
            return redirected

    return p_resolved


@dataclass
class ApplicationConfig:
    name: str = "KYC Document Sorter"
    version: str = "2.0.0"


@dataclass
class InputConfig:
    directory: str = "Source Files/Input"
    resolved_path: Path = field(init=False)

    def resolve(self, base_dir: Path, allow_legacy_paths: bool = False) -> Path:
        if not self.directory or not str(self.directory).strip():
            raise ConfigurationError("Input directory cannot be empty.")
        p = Path(self.directory)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (base_dir / p).resolve()
        self.resolved_path = _check_forbidden_paths(resolved, "input", allow_legacy_paths=allow_legacy_paths)
        return self.resolved_path


@dataclass
class OutputConfig:
    directory: str = "Destination files"
    operation: str = "move"  # "copy" or "move"
    create_date_folder: bool = False
    date_format: str = "%Y-%m-%d"
    kyc_folder: str = "KYC"
    unknown_folder: str = "NON KYC"
    unidentified_folder: str = "Unidentified_files"
    generate_companion_files: bool = True
    resolved_path: Path = field(init=False)

    def resolve(self, base_dir: Path, allow_legacy_paths: bool = False) -> Path:
        if not self.directory or not str(self.directory).strip():
            raise ConfigurationError("Output directory cannot be empty.")
        p = Path(self.directory)
        if p.is_absolute():
            resolved = p.resolve()
        else:
            resolved = (base_dir / p).resolve()
        self.resolved_path = _check_forbidden_paths(resolved, "output", allow_legacy_paths=allow_legacy_paths)
        return self.resolved_path


@dataclass
class VLMConfig:
    model: str = "qwen3-vl:4b-instruct"
    endpoint: str = "http://localhost:11434/api/chat"
    timeout: int = 300
    max_retries: int = 3
    temperature: float = 0.0
    num_predict: int = 50
    num_ctx: int = 8192
    single_pass_mode: bool = True
    image_max_dimension: int = 1200
    image_quality: int = 85
    keep_alive: Union[str, int] = "24h"


@dataclass
class ProcessingConfig:
    confidence_threshold: float = 0.80
    dry_run: bool = False
    max_workers: int = 4
    cpu_limit_percent: float = 80.0
    throttle_cpu_percent: float = 75.0
    adaptive_workers: bool = True
    multiple_document_score_margin: float = 0.10
    limit: Optional[int] = None
    batch_size: int = 100
    journal_path: Optional[str] = None
    watcher_max_queue: int = 1000
    supported_extensions: Set[str] = field(
        default_factory=lambda: {".jpg"}
    )


@dataclass
class WatcherConfig:
    stability_delay_sec: float = 1.0
    settle_timeout_sec: float = 30.0
    poll_interval_sec: float = 0.5
    use_polling: bool = False
    polling_interval_sec: float = 5.0


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
class SecurityConfig:
    log_sensitive_text: bool = False


@dataclass
class Config:
    application: ApplicationConfig = field(default_factory=ApplicationConfig)
    input: InputConfig = field(default_factory=InputConfig)
    output: OutputConfig = field(default_factory=OutputConfig)
    vlm: VLMConfig = field(default_factory=VLMConfig)
    processing: ProcessingConfig = field(default_factory=ProcessingConfig)
    watcher: WatcherConfig = field(default_factory=WatcherConfig)
    ocr: Optional[Any] = None
    security: SecurityConfig = field(default_factory=SecurityConfig)
    journal_path: Optional[str] = None
    kyc_folder: str = "KYC"
    unknown_folder: str = "NON KYC"
    unidentified_folder: str = "Unidentified_files"
    unknown_suffix: str = "UNKNOWN"
    documents: Dict[str, DocumentTypeMetadata] = field(default_factory=lambda: dict(DEFAULT_DOCUMENT_METADATA))
    folders: Dict[str, str] = field(
        default_factory=lambda: {k: v.folder for k, v in DEFAULT_DOCUMENT_METADATA.items()}
    )
    suffixes: Dict[str, str] = field(
        default_factory=lambda: {k: v.filename_suffix for k, v in DEFAULT_DOCUMENT_METADATA.items()}
    )
    base_dir: Path = field(default_factory=lambda: Path.cwd())

    @property
    def input_dir(self) -> Path:
        """Configured source input directory."""
        return self.input.resolved_path

    @property
    def kyc_dir(self) -> Path:
        """Destination directory for verified KYC documents."""
        return self.output.resolved_path / self.kyc_folder

    @property
    def non_kyc_dir(self) -> Path:
        """Destination directory for classified NON_KYC documents."""
        return self.output.resolved_path / self.unknown_folder

    @property
    def unidentified_dir(self) -> Path:
        """Destination directory for unclassified / failed documents."""
        return self.output.resolved_path / self.unidentified_folder

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


def _parse_bool(val: Any) -> bool:
    """Helper to parse boolean values from environment variables or configs."""
    if isinstance(val, bool):
        return val
    s = str(val).strip().lower()
    return s in ("1", "true", "yes", "on", "enable", "enabled")


def _apply_env_overrides(raw_data: Dict[str, Any]) -> None:
    """
    Applies environment variable overrides to configuration data with standard precedence.
    Supports both KYC_* prefixed variables and standard OLLAMA_* variables.
    """
    env = os.environ

    # Input directory
    in_dir = env.get("KYC_INPUT_DIR") or env.get("KYC_INPUT_DIRECTORY")
    if in_dir:
        raw_data.setdefault("input", {})["directory"] = in_dir

    # Output directory
    out_dir = env.get("KYC_OUTPUT_DIR") or env.get("KYC_OUTPUT_DIRECTORY")
    if out_dir:
        raw_data.setdefault("output", {})["directory"] = out_dir

    # Output operation
    op = env.get("KYC_OPERATION")
    if op:
        raw_data.setdefault("output", {})["operation"] = op

    # Create date folder
    if "KYC_CREATE_DATE_FOLDER" in env:
        raw_data.setdefault("output", {})["create_date_folder"] = _parse_bool(env["KYC_CREATE_DATE_FOLDER"])

    # Processing confidence threshold
    conf = env.get("KYC_CONFIDENCE_THRESHOLD")
    if conf:
        try:
            raw_data.setdefault("processing", {})["confidence_threshold"] = float(conf)
        except ValueError:
            raise ConfigurationError(f"Invalid KYC_CONFIDENCE_THRESHOLD environment variable: '{conf}'")

    # Dry run
    if "KYC_DRY_RUN" in env:
        raw_data.setdefault("processing", {})["dry_run"] = _parse_bool(env["KYC_DRY_RUN"])

    # Max workers
    workers = env.get("KYC_MAX_WORKERS") or env.get("KYC_WORKERS")
    if workers:
        try:
            raw_data.setdefault("processing", {})["max_workers"] = int(workers)
        except ValueError:
            raise ConfigurationError(f"Invalid worker count environment variable: '{workers}'")

    # CPU thresholds
    cpu_limit = env.get("KYC_CPU_LIMIT_PERCENT")
    if cpu_limit:
        try:
            raw_data.setdefault("processing", {})["cpu_limit_percent"] = float(cpu_limit)
        except ValueError:
            raise ConfigurationError(f"Invalid KYC_CPU_LIMIT_PERCENT: '{cpu_limit}'")

    cpu_throttle = env.get("KYC_CPU_THROTTLE_PERCENT")
    if cpu_throttle:
        try:
            raw_data.setdefault("processing", {})["throttle_cpu_percent"] = float(cpu_throttle)
        except ValueError:
            raise ConfigurationError(f"Invalid KYC_CPU_THROTTLE_PERCENT: '{cpu_throttle}'")

    # Limit
    limit = env.get("KYC_LIMIT")
    if limit:
        try:
            raw_data.setdefault("processing", {})["limit"] = int(limit)
        except ValueError:
            raise ConfigurationError(f"Invalid KYC_LIMIT: '{limit}'")

    # Supported extensions
    exts = env.get("KYC_SUPPORTED_EXTENSIONS")
    if exts:
        raw_data.setdefault("processing", {})["supported_extensions"] = [
            e.strip() for e in exts.split(",") if e.strip()
        ]

    # Bounded batch size
    batch_sz = env.get("KYC_BATCH_SIZE")
    if batch_sz:
        try:
            raw_data.setdefault("processing", {})["batch_size"] = int(batch_sz)
        except ValueError:
            raise ConfigurationError(f"Invalid KYC_BATCH_SIZE: '{batch_sz}'")

    # SQLite journal path
    j_path = env.get("KYC_JOURNAL_PATH") or env.get("KYC_JOURNAL")
    if j_path:
        raw_data.setdefault("processing", {})["journal_path"] = j_path
        raw_data["journal_path"] = j_path

    # Watcher max queue
    w_queue = env.get("KYC_WATCHER_MAX_QUEUE")
    if w_queue:
        try:
            raw_data.setdefault("processing", {})["watcher_max_queue"] = int(w_queue)
        except ValueError:
            raise ConfigurationError(f"Invalid KYC_WATCHER_MAX_QUEUE: '{w_queue}'")

    # Ollama / VLM Endpoint
    endpoint = env.get("KYC_VLM_ENDPOINT") or env.get("OLLAMA_ENDPOINT") or env.get("OLLAMA_HOST")
    if endpoint:
        raw_data.setdefault("vlm", {})["endpoint"] = endpoint

    # Ollama / VLM Model
    model = env.get("KYC_VLM_MODEL") or env.get("OLLAMA_MODEL")
    if model:
        raw_data.setdefault("vlm", {})["model"] = model

    # Ollama / VLM Timeout
    timeout = env.get("KYC_VLM_TIMEOUT") or env.get("OLLAMA_TIMEOUT")
    if timeout:
        try:
            raw_data.setdefault("vlm", {})["timeout"] = int(timeout)
        except ValueError:
            raise ConfigurationError(f"Invalid VLM timeout environment variable: '{timeout}'")

    # Ollama / VLM Max Retries
    retries = env.get("KYC_VLM_MAX_RETRIES") or env.get("KYC_MAX_RETRIES")
    if retries:
        try:
            raw_data.setdefault("vlm", {})["max_retries"] = int(retries)
        except ValueError:
            raise ConfigurationError(f"Invalid VLM retries: '{retries}'")

    if "KYC_SINGLE_PASS_MODE" in env:
        raw_data.setdefault("vlm", {})["single_pass_mode"] = _parse_bool(env["KYC_SINGLE_PASS_MODE"])

    # Ollama / VLM Keep-Alive
    keep_alive = env.get("KYC_VLM_KEEP_ALIVE") or env.get("OLLAMA_KEEP_ALIVE")
    if keep_alive is not None:
        raw_data.setdefault("vlm", {})["keep_alive"] = keep_alive

    # Ollama / VLM Context Length
    num_ctx = env.get("KYC_VLM_NUM_CTX") or env.get("OLLAMA_NUM_CTX")
    if num_ctx is not None:
        try:
            raw_data.setdefault("vlm", {})["num_ctx"] = int(num_ctx)
        except ValueError:
            raise ConfigurationError(f"Invalid KYC_VLM_NUM_CTX: '{num_ctx}'")

    if "KYC_IMAGE_MAX_DIMENSION" in env:
        try:
            raw_data.setdefault("vlm", {})["image_max_dimension"] = int(env["KYC_IMAGE_MAX_DIMENSION"])
        except ValueError:
            raise ConfigurationError(f"Invalid KYC_IMAGE_MAX_DIMENSION: '{env['KYC_IMAGE_MAX_DIMENSION']}'")

    if "KYC_IMAGE_QUALITY" in env:
        try:
            raw_data.setdefault("vlm", {})["image_quality"] = int(env["KYC_IMAGE_QUALITY"])
        except ValueError:
            raise ConfigurationError(f"Invalid KYC_IMAGE_QUALITY: '{env['KYC_IMAGE_QUALITY']}'")

    # Watcher settings
    if "KYC_WATCHER_STABILITY_DELAY" in env:
        try:
            raw_data.setdefault("watcher", {})["stability_delay_sec"] = float(env["KYC_WATCHER_STABILITY_DELAY"])
        except ValueError:
            raise ConfigurationError(f"Invalid KYC_WATCHER_STABILITY_DELAY: '{env['KYC_WATCHER_STABILITY_DELAY']}'")

    if "KYC_WATCHER_SETTLE_TIMEOUT" in env:
        try:
            raw_data.setdefault("watcher", {})["settle_timeout_sec"] = float(env["KYC_WATCHER_SETTLE_TIMEOUT"])
        except ValueError:
            raise ConfigurationError(f"Invalid KYC_WATCHER_SETTLE_TIMEOUT: '{env['KYC_WATCHER_SETTLE_TIMEOUT']}'")

    # Polling fallback for mounts that do not deliver filesystem events
    if "KYC_WATCHER_USE_POLLING" in env:
        raw_data.setdefault("watcher", {})["use_polling"] = _parse_bool(env["KYC_WATCHER_USE_POLLING"])

    if "KYC_WATCHER_POLLING_INTERVAL" in env:
        try:
            raw_data.setdefault("watcher", {})["polling_interval_sec"] = float(env["KYC_WATCHER_POLLING_INTERVAL"])
        except ValueError:
            raise ConfigurationError(f"Invalid KYC_WATCHER_POLLING_INTERVAL: '{env['KYC_WATCHER_POLLING_INTERVAL']}'")

    # Security settings
    if "KYC_LOG_SENSITIVE_TEXT" in env:
        raw_data.setdefault("security", {})["log_sensitive_text"] = _parse_bool(env["KYC_LOG_SENSITIVE_TEXT"])


def validate_config(config: Config) -> None:
    """
    Validates configuration settings at startup.
    Fails early with informative ConfigurationError messages for invalid configurations.
    """
    # 1. Input and Output directory validation
    if not config.input.directory or not str(config.input.directory).strip():
        raise ConfigurationError("Input directory setting cannot be empty.")
    if not config.output.directory or not str(config.output.directory).strip():
        raise ConfigurationError("Output directory setting cannot be empty.")

    in_p = config.input.resolved_path
    out_p = config.output.resolved_path

    # Directory conflict detection
    if in_p == out_p:
        raise ConfigurationError(
            f"Input directory and Output directory cannot be the same directory ('{in_p}'). "
            "This would cause infinite loops and overwrite source files."
        )
    if in_p in out_p.parents:
        raise ConfigurationError(
            f"Output directory ('{out_p}') cannot be inside Input directory ('{in_p}'). "
            "This would cause segregated output files to be detected as incoming files and reprocessed."
        )
    if out_p in in_p.parents:
        raise ConfigurationError(
            f"Input directory ('{in_p}') cannot be inside Output directory ('{out_p}'). "
            "This would create path conflicts during segregation."
        )

    # 2. Output operation
    if config.output.operation not in ("copy", "move"):
        raise ConfigurationError(
            f"Invalid file operation '{config.output.operation}'. Must be either 'copy' or 'move'."
        )

    # 3. Processing settings
    if config.processing.max_workers < 1:
        raise ConfigurationError(
            f"Invalid worker count ({config.processing.max_workers}). Must be at least 1."
        )
    if not (0.0 <= config.processing.confidence_threshold <= 1.0):
        raise ConfigurationError(
            f"Invalid confidence threshold ({config.processing.confidence_threshold}). Must be between 0.0 and 1.0."
        )
    if not (0.0 < config.processing.cpu_limit_percent <= 100.0):
        raise ConfigurationError(
            f"Invalid cpu_limit_percent ({config.processing.cpu_limit_percent}). Must be between 1.0 and 100.0."
        )
    if not (0.0 < config.processing.throttle_cpu_percent <= 100.0):
        raise ConfigurationError(
            f"Invalid throttle_cpu_percent ({config.processing.throttle_cpu_percent}). Must be between 1.0 and 100.0."
        )
    if config.processing.throttle_cpu_percent > config.processing.cpu_limit_percent:
        raise ConfigurationError(
            f"Throttle threshold ({config.processing.throttle_cpu_percent}%) cannot exceed "
            f"CPU ceiling limit ({config.processing.cpu_limit_percent}%)."
        )
    if config.processing.batch_size <= 0:
        raise ConfigurationError(
            f"Invalid batch_size ({config.processing.batch_size}). Must be greater than 0."
        )

    # 4. VLM settings
    if not config.vlm.model or not config.vlm.model.strip():
        raise ConfigurationError("VLM model name cannot be empty.")
    if config.vlm.timeout <= 0:
        raise ConfigurationError(
            f"Invalid VLM request timeout ({config.vlm.timeout}). Must be greater than 0 seconds."
        )
    if config.vlm.max_retries < 0:
        raise ConfigurationError(
            f"Invalid VLM max_retries ({config.vlm.max_retries}). Cannot be negative."
        )
    # Validate endpoint URL formatting
    _ = normalize_ollama_endpoint(config.vlm.endpoint)
    if config.vlm.image_max_dimension < 100:
        raise ConfigurationError(
            f"Invalid VLM image_max_dimension ({config.vlm.image_max_dimension}). Must be >= 100 pixels."
        )
    if not (1 <= config.vlm.image_quality <= 100):
        raise ConfigurationError(
            f"Invalid VLM image_quality ({config.vlm.image_quality}). Must be between 1 and 100."
        )

    # 5. Watcher settings
    if config.watcher.stability_delay_sec <= 0:
        raise ConfigurationError("Watcher stability_delay_sec must be greater than 0.")
    if config.watcher.settle_timeout_sec <= 0:
        raise ConfigurationError("Watcher settle_timeout_sec must be greater than 0.")
    if config.watcher.poll_interval_sec <= 0:
        raise ConfigurationError("Watcher poll_interval_sec must be greater than 0.")
    if config.watcher.polling_interval_sec <= 0:
        raise ConfigurationError("Watcher polling_interval_sec must be greater than 0.")


def load_config(
    config_path: Optional[Union[str, Path]] = None,
    overrides: Optional[Dict[str, Any]] = None,
    base_dir: Optional[Union[str, Path]] = None,
    allow_legacy_paths: bool = False,
    validate: bool = True,
) -> Config:
    """
    Load configuration with documented precedence contract:
      1. Runtime overrides (e.g. CLI flags)
      2. Environment variables (KYC_*, OLLAMA_*)
      3. YAML configuration file
      4. Default values

    Args:
        config_path: Optional path to config.yaml. Looks in standard locations if None.
        overrides: Runtime overrides dictionary (highest priority).
        base_dir: Base directory for relative path resolution (defaults to config root).
        allow_legacy_paths: When True, safely redirects legacy paths instead of failing.
        validate: When True, validates all configuration parameters at startup.

    Returns:
        Config: Populated and validated configuration object.
    """
    raw_data: Dict[str, Any] = {}
    actual_path: Optional[Path] = None
    project_root = Path(__file__).resolve().parent.parent.parent

    if config_path:
        p = Path(config_path)
        if not p.is_file():
            raise ConfigurationError(f"Specified configuration file does not exist: '{config_path}'")
        actual_path = p.resolve()
    else:
        candidates = [
            project_root / "config.yaml",
            Path(__file__).resolve().parent.parent / "config.yaml",
            Path.cwd() / "config.yaml",
        ]
        for c in candidates:
            if c.is_file():
                actual_path = c.resolve()
                break

    if actual_path and actual_path.is_file():
        try:
            with open(actual_path, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if isinstance(loaded, dict):
                    raw_data = loaded
        except yaml.YAMLError as exc:
            raise ConfigurationError(f"Malformed YAML in configuration file '{actual_path}': {exc}") from exc

    # Apply environment variable overrides (precedence 2)
    _apply_env_overrides(raw_data)

    # Apply explicit parameter overrides (precedence 1: highest)
    if overrides:
        _deep_update(raw_data, overrides)

    # Determine base directory for relative path resolution:
    # 1. Explicit base_dir if supplied
    # 2. Directory of the loaded config file
    # 3. Project root directory
    if base_dir:
        resolved_base = Path(base_dir).resolve()
    elif actual_path:
        resolved_base = actual_path.parent.resolve()
    else:
        resolved_base = project_root.resolve()

    # Application section
    app_data = raw_data.get("application", {})
    app_cfg = ApplicationConfig(
        name=str(app_data.get("name", "KYC Document Sorter")),
        version=str(app_data.get("version", "2.0.0")),
    )

    # Input section
    in_data = raw_data.get("input", {})
    in_dir_str = str(in_data.get("directory", "Source Files/Input")).strip()
    in_cfg = InputConfig(directory=in_dir_str)
    in_cfg.resolve(resolved_base, allow_legacy_paths=allow_legacy_paths)

    # Output section
    out_data = raw_data.get("output", {})
    op_raw = str(out_data.get("operation", "copy")).strip().lower()
    op = op_raw if op_raw in ("copy", "move") else "copy"
    kyc_f = str(out_data.get("kyc_folder", "KYC")).strip()
    unidentified_f = str(out_data.get("unidentified_folder", "Unidentified_files")).strip()
    out_dir_str = str(out_data.get("directory", "Destination files")).strip()
    out_cfg = OutputConfig(
        directory=out_dir_str,
        operation=op,
        create_date_folder=_parse_bool(out_data.get("create_date_folder", False)),
        date_format=str(out_data.get("date_format", "%Y-%m-%d")),
        kyc_folder=kyc_f,
        unknown_folder=str(out_data.get("unknown_folder", "NON KYC")).strip(),
        unidentified_folder=unidentified_f,
        generate_companion_files=_parse_bool(out_data.get("generate_companion_files", True)),
    )
    out_cfg.resolve(resolved_base, allow_legacy_paths=allow_legacy_paths)

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
        max_workers = int(proc_data.get("max_workers", 4))
    except (TypeError, ValueError):
        max_workers = 4

    try:
        cpu_limit = float(proc_data.get("cpu_limit_percent", 80.0))
    except (TypeError, ValueError):
        cpu_limit = 80.0

    try:
        throttle_cpu = float(proc_data.get("throttle_cpu_percent", 75.0))
    except (TypeError, ValueError):
        throttle_cpu = 75.0

    adaptive_workers = _parse_bool(proc_data.get("adaptive_workers", True))

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

    batch_size_val = int(proc_data.get("batch_size", 100))
    journal_p = proc_data.get("journal_path") or raw_data.get("journal_path")
    w_queue = int(proc_data.get("watcher_max_queue", 1000))

    proc_cfg = ProcessingConfig(
        confidence_threshold=conf_thresh,
        dry_run=_parse_bool(proc_data.get("dry_run", False)),
        max_workers=max_workers,
        cpu_limit_percent=cpu_limit,
        throttle_cpu_percent=throttle_cpu,
        adaptive_workers=adaptive_workers,
        multiple_document_score_margin=score_margin,
        limit=limit_val,
        batch_size=batch_size_val,
        journal_path=journal_p,
        watcher_max_queue=w_queue,
        supported_extensions=clean_exts,
    )

    # VLM section
    vlm_data = raw_data.get("vlm", {})
    raw_endpoint = str(vlm_data.get("endpoint", "http://localhost:11434/api/chat")).strip()
    try:
        norm_endpoint = normalize_ollama_endpoint(raw_endpoint)
    except ConfigurationError:
        if validate:
            raise
        norm_endpoint = raw_endpoint

    try:
        timeout_val = int(vlm_data.get("timeout", 300))
    except (TypeError, ValueError):
        timeout_val = 300

    vlm_cfg = VLMConfig(
        model=str(vlm_data.get("model", "qwen3-vl:4b-instruct")).strip(),
        endpoint=norm_endpoint,
        timeout=timeout_val,
        max_retries=int(vlm_data.get("max_retries", 3)),
        temperature=float(vlm_data.get("temperature", 0.0)),
        num_predict=int(vlm_data.get("num_predict", 50)),
        num_ctx=int(vlm_data.get("num_ctx", 8192)),
        single_pass_mode=_parse_bool(vlm_data.get("single_pass_mode", True)),
        image_max_dimension=int(vlm_data.get("image_max_dimension", 1200)),
        image_quality=int(vlm_data.get("image_quality", 85)),
        keep_alive=vlm_data.get("keep_alive", "24h"),
    )

    # Watcher section
    watch_data = raw_data.get("watcher", {})
    watcher_cfg = WatcherConfig(
        stability_delay_sec=float(watch_data.get("stability_delay_sec", 1.0)),
        settle_timeout_sec=float(watch_data.get("settle_timeout_sec", 30.0)),
        poll_interval_sec=float(watch_data.get("poll_interval_sec", 0.5)),
        use_polling=_parse_bool(watch_data.get("use_polling", False)),
        polling_interval_sec=float(watch_data.get("polling_interval_sec", 5.0)),
    )

    # Security section
    sec_data = raw_data.get("security", {})
    sec_cfg = SecurityConfig(
        log_sensitive_text=_parse_bool(sec_data.get("log_sensitive_text", False)),
    )

    # Unknown / NON KYC settings
    unk_data = raw_data.get("unknown", {})
    unk_folder = str(unk_data.get("folder", out_data.get("unknown_folder", "NON KYC"))).strip()
    unk_suffix = str(unk_data.get("filename_suffix", "UNKNOWN")).strip()

    # Document type mappings (folders and filename suffixes)
    docs_map: Dict[str, DocumentTypeMetadata] = dict(DEFAULT_DOCUMENT_METADATA)
    folders_map: Dict[str, str] = {k: v.folder for k, v in DEFAULT_DOCUMENT_METADATA.items()}
    suffixes_map: Dict[str, str] = {k: v.filename_suffix for k, v in DEFAULT_DOCUMENT_METADATA.items()}

    custom_docs = raw_data.get("documents", {})
    if isinstance(custom_docs, dict):
        for raw_k, meta in custom_docs.items():
            k = str(raw_k).strip().upper()
            if isinstance(meta, dict):
                f_name = str(meta.get("folder", folders_map.get(k, k)))
                s_name = str(meta.get("filename_suffix", suffixes_map.get(k, k)))
                docs_map[k] = DocumentTypeMetadata(folder=f_name, filename_suffix=s_name)

    if "UNKNOWN" not in folders_map:
        folders_map["UNKNOWN"] = unk_folder
    if "UNKNOWN" not in suffixes_map:
        suffixes_map["UNKNOWN"] = unk_suffix
    if "UNKNOWN" not in docs_map:
        docs_map["UNKNOWN"] = DocumentTypeMetadata(folder=unk_folder, filename_suffix=unk_suffix)

    cfg = Config(
        application=app_cfg,
        input=in_cfg,
        output=out_cfg,
        vlm=vlm_cfg,
        processing=proc_cfg,
        watcher=watcher_cfg,
        ocr=None,
        security=sec_cfg,
        journal_path=journal_p,
        kyc_folder=kyc_f,
        unknown_folder=unk_folder,
        unidentified_folder=unidentified_f,
        unknown_suffix=unk_suffix,
        documents=docs_map,
        folders=folders_map,
        suffixes=suffixes_map,
        base_dir=resolved_base,
    )

    if validate:
        validate_config(cfg)

    return cfg
