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

from kyc_sorter.detector.classifier import DocumentClassifier
from kyc_sorter.detector.ocr import OCREngine
from kyc_sorter.sorter.batch_processor import BatchProcessor
from kyc_sorter.sorter.file_router import FileRouter
from kyc_sorter.utils.config_loader import load_config
from kyc_sorter.utils.logger import setup_logger


def parse_args() -> argparse.Namespace:
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
        "--gui",
        action="store_true",
        help="Launch the graphical desktop interface.",
    )
    return parser.parse_args()


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

    config_path = Path(args.config) if args.config else None
    config = load_config(config_path=config_path, overrides=overrides)

    logger = setup_logger(
        name="kyc_sorter",
        log_dir=config.output.resolved_path.parent / "logs",
        mask_sensitive=not config.security.log_sensitive_text,
    )

    print("=" * 60)
    print(f" {config.application.name} v{config.application.version}")
    print("=" * 60)
    print(f"Input Directory      : {config.input.resolved_path}")
    print(f"Output Directory     : {config.output.resolved_path}")
    print(f"Execution Mode       : {'SOURCE FOLDER EVENT WATCHER' if getattr(args, 'watch', False) else 'BATCH PROCESSING'}")
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
