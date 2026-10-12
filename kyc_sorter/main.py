"""
KYC Document Sorter — Main Application Entry Point.
Executes batch document classification and segregation via CLI or GUI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Ensure package directory is on sys.path
_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent
for p in (str(_current_dir), str(_parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from kyc_sorter.sorter.batch_processor import BatchProcessor
from kyc_sorter.sorter.file_router import FileRouter
from kyc_sorter.utils.config_loader import load_config
from kyc_sorter.utils.logger import setup_logger


def parse_args(argv: Optional[List[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="KYC Document Sorter — Automated multi-signal document identification & segregation."
    )
    parser.add_argument(
        "-i", "--input",
        type=str,
        default=None,
        help="Path to source directory containing incoming KYC documents.",
    )
    parser.add_argument(
        "-o", "--output",
        type=str,
        default=None,
        help="Path to destination directory for segregated output.",
    )
    parser.add_argument(
        "--operation",
        type=str,
        choices=["copy", "move"],
        default=None,
        help="File operation mode: 'copy' (default, safe) or 'move'.",
    )
    parser.add_argument(
        "-c", "--confidence",
        type=float,
        default=None,
        help="Confidence threshold between 0.0 and 1.0 (default: 0.80).",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        default=None,
        help="Simulate classification and routing without modifying any files.",
    )
    parser.add_argument(
        "-w", "--watch",
        action="store_true",
        help="Run continuous filesystem event watcher on source directory for incoming files.",
    )
    parser.add_argument(
        "--process-existing",
        action="store_true",
        help="When running in watch mode, also process existing files in source directory on startup.",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to custom config.yaml file.",
    )
    parser.add_argument(
        "-n", "--limit",
        type=int,
        default=None,
        help="Limit number of documents to process in batch mode.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=None,
        help="Number of concurrent worker threads for VLM processing (default: 4).",
    )
    parser.add_argument(
        "--ollama-endpoint", "--endpoint",
        type=str,
        default=None,
        help="Ollama API endpoint URL (e.g. http://localhost:11434/api/chat or http://ollama:11434).",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Vision-Language Model name (default: qwen3-vl:4b-instruct).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=None,
        help="VLM request timeout in seconds (default: 300).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Chunk batch size for bounded incremental processing (default: 100).",
    )
    parser.add_argument(
        "--journal",
        type=str,
        default=None,
        help="Custom path to SQLite recovery journal file.",
    )
    parser.add_argument(
        "--retries",
        type=int,
        default=None,
        help="Max retries for transient Ollama errors (default: 3).",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch the graphical desktop interface.",
    )
    parser.add_argument(
        "--check-connection",
        action="store_true",
        help="Verify connection to Ollama API and confirm model availability, then exit.",
    )
    parser.add_argument(
        "--single-pass",
        action="store_true",
        default=None,
        help="Enable single-pass classification (unifies multi-doc detection into master classification for 2.7x speedup).",
    )
    parser.add_argument(
        "--two-pass",
        action="store_true",
        default=None,
        help="Force two-pass classification (separate multi-document detection call).",
    )
    parser.add_argument(
        "--keep-alive",
        type=str,
        default=None,
        help="Ollama model keep-alive duration in VRAM (e.g. '24h', '30m', '0', '-1').",
    )

    sanitized_argv = list(argv) if argv is not None else list(sys.argv[1:])
    while sanitized_argv and sanitized_argv[0] in ("python", "python3", "main.py", "./main.py"):
        sanitized_argv.pop(0)
    return parser.parse_args(sanitized_argv)


def run_cli(args: argparse.Namespace) -> int:
    # Build dictionary of CLI overrides
    overrides: dict = {}
    if args.input:
        overrides.setdefault("input", {})["directory"] = str(Path(args.input).resolve())
    if args.output:
        overrides.setdefault("output", {})["directory"] = str(Path(args.output).resolve())
    if args.operation:
        overrides.setdefault("output", {})["operation"] = args.operation
    if args.confidence is not None:
        overrides.setdefault("processing", {})["confidence_threshold"] = args.confidence
    if args.dry_run is not None and args.dry_run:
        overrides.setdefault("processing", {})["dry_run"] = True
    if args.limit is not None:
        overrides.setdefault("processing", {})["limit"] = args.limit
    if getattr(args, "workers", None) is not None:
        overrides.setdefault("processing", {})["max_workers"] = args.workers
    if getattr(args, "ollama_endpoint", None):
        overrides.setdefault("vlm", {})["endpoint"] = args.ollama_endpoint
    if getattr(args, "model", None):
        overrides.setdefault("vlm", {})["model"] = args.model
    if getattr(args, "timeout", None) is not None:
        overrides.setdefault("vlm", {})["timeout"] = args.timeout
    if getattr(args, "batch_size", None) is not None:
        overrides.setdefault("processing", {})["batch_size"] = args.batch_size
    if getattr(args, "journal", None):
        overrides.setdefault("processing", {})["journal_path"] = args.journal
    if getattr(args, "retries", None) is not None:
        overrides.setdefault("vlm", {})["max_retries"] = args.retries
    if getattr(args, "keep_alive", None) is not None:
        overrides.setdefault("vlm", {})["keep_alive"] = args.keep_alive
    if getattr(args, "single_pass", None):
        overrides.setdefault("vlm", {})["single_pass_mode"] = True
    elif getattr(args, "two_pass", None):
        overrides.setdefault("vlm", {})["single_pass_mode"] = False

    config_path = Path(args.config) if args.config else None
    from kyc_sorter.utils.config_loader import ConfigurationError
    try:
        config = load_config(config_path=config_path, overrides=overrides)
    except ConfigurationError as err:
        print(f"CONFIGURATION ERROR: {err}", file=sys.stderr)
        return 1

    # Optional preflight connectivity check
    if getattr(args, "check_connection", False):
        from kyc_sorter.utils.healthcheck import verify_ollama_connectivity
        print(f"Testing connection to Ollama API at '{config.vlm.endpoint}' for model '{config.vlm.model}'...")
        is_healthy, msg = verify_ollama_connectivity(
            endpoint=config.vlm.endpoint,
            model_name=config.vlm.model,
            timeout=min(getattr(config.vlm, "timeout", 10), 15),
        )
        if is_healthy:
            print(f"[SUCCESS] {msg}")
            return 0
        else:
            print(f"[FAILURE] {msg}", file=sys.stderr)
            return 1

    import os
    custom_log_dir = os.environ.get("KYC_LOG_DIR")
    log_dir_path = Path(custom_log_dir) if custom_log_dir else config.output.resolved_path.parent / "logs"

    logger = setup_logger(
        name="kyc_sorter",
        log_dir=log_dir_path,
        mask_sensitive=not config.security.log_sensitive_text,
    )

    workers_display = getattr(config.processing, "max_workers", 4)
    batch_size_display = getattr(config.processing, "batch_size", 100)
    journal_display = getattr(config.processing, "journal_path", None) or "(default: destination/kyc_journal.db)"
    print("=" * 60)
    print(f" {config.application.name} v{config.application.version} (VLM Engine)")
    print("=" * 60)
    print(f"Input Directory      : {config.input.resolved_path}")
    print(f"Output Directory     : {config.output.resolved_path}")
    print(f"Execution Mode       : {'SOURCE FOLDER EVENT WATCHER' if getattr(args, 'watch', False) else 'BATCH PROCESSING'}")
    print(f"Pipeline Engine      : VLM ({config.vlm.model} via Ollama)")
    print(f"Ollama Endpoint      : {config.vlm.endpoint}")
    print(f"Request Timeout      : {config.vlm.timeout}s")
    print(f"Target Workers       : {workers_display}")
    print(f"Batch Chunk Size     : {batch_size_display}")
    print(f"SQLite Journal DB    : {journal_display}")
    print(f"Max Request Retries  : {config.vlm.max_retries}")
    print(f"Operation Mode       : {config.output.operation.upper()}")
    print(f"Dry Run (Test Mode)  : {'ENABLED' if config.processing.dry_run else 'DISABLED'}")
    print(f"Confidence Threshold : {config.processing.confidence_threshold:.0%}")
    print(f"Date Partitioning    : {'ENABLED' if config.output.create_date_folder else 'DISABLED'}")
    print("=" * 60)

    if not config.input.resolved_path.is_dir():
        print(f"ERROR: Input directory does not exist: {config.input.resolved_path}", file=sys.stderr)
        return 1

    if getattr(args, "watch", False):
        from kyc_sorter.sorter.watcher import SourceFolderWatcher
        watcher = SourceFolderWatcher(config=config)
        try:
            watcher.start(process_existing=bool(getattr(args, "process_existing", False)))
            print("\nSource folder event watcher is active. Press Ctrl+C to terminate...\n")
            while watcher.is_running:
                import time
                time.sleep(1.0)
        except KeyboardInterrupt:
            print("\nTermination signal received. Stopping watcher...")
        finally:
            watcher.stop()
        return 0

    processor = BatchProcessor(config=config)
    summary = processor.process_batch(limit=args.limit)
    summary.print_summary()

    return 0


def main() -> int:
    args = parse_args()
    if args.gui:
        from kyc_sorter.gui import launch_gui
        launch_gui(config_path=Path(args.config) if args.config else None)
        return 0
    return run_cli(args)


if __name__ == "__main__":
    sys.exit(main())
