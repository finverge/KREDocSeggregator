"""
Logging setup for KYC Document Sorter.
Provides sanitized logging with timestamped file handlers and console output.
"""

from __future__ import annotations

import datetime
import logging
import re
import sys
from pathlib import Path
from typing import Optional


class SensitiveDataFilter(logging.Filter):
    """Filter that masks sensitive 10-12 digit numbers and identifiers from log records."""

    def __init__(self, mask_enabled: bool = True):
        super().__init__()
        self.mask_enabled = mask_enabled

    def filter(self, record: logging.LogRecord) -> bool:
        if not self.mask_enabled:
            return True
        if isinstance(record.msg, str):
            record.msg = self.mask_text(record.msg)
        return True

    @staticmethod
    def mask_text(text: str) -> str:
        # Mask 12-digit Aadhaar pattern
        text = re.sub(r"\b\d{4}\s?\d{4}\s?\d{4}\b", "XXXX-XXXX-XXXX", text)
        # Mask 10-char PAN pattern
        text = re.sub(r"\b[A-Z]{5}\d{4}[A-Z]\b", "XXXXX0000X", text)
        # Mask 9-18 digit account numbers
        text = re.sub(r"\b\d{9,18}\b", "XXXXXXXXXX", text)
        return text


def setup_logger(
    name: str = "kyc_sorter",
    log_dir: Optional[Path] = None,
    log_to_console: bool = True,
    log_to_file: bool = True,
    level: int = logging.INFO,
    mask_sensitive: bool = True,
) -> logging.Logger:
    """Configure and return an application-wide logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    # Avoid duplicate handlers if already configured
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="[%(asctime)s] [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    sensitive_filter = SensitiveDataFilter(mask_enabled=mask_sensitive)

    if log_to_console:
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(formatter)
        ch.addFilter(sensitive_filter)
        logger.addHandler(ch)

    if log_to_file:
        if log_dir is None:
            log_dir = Path("logs")
        log_dir = Path(log_dir)
        log_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H%M%S")
        log_file = log_dir / f"processing_{timestamp}.log"
        fh = logging.FileHandler(str(log_file), encoding="utf-8")
        fh.setLevel(level)
        fh.setFormatter(formatter)
        fh.addFilter(sensitive_filter)
        logger.addHandler(fh)

    return logger
