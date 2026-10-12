"""
Persistent SQLite-based processing journal for KYC Document Sorter.
Provides ACID transaction logging, crash recovery reconciliation, and safe resume.
"""

from __future__ import annotations

import hashlib
import logging
import os
from pathlib import Path
import sqlite3
import threading
import time
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)


def compute_quick_hash(file_path: Path, chunk_size: int = 65536) -> str:
    """Computes SHA-256 hash prefix (first 64KB + length) for fast file identity verification."""
    try:
        hasher = hashlib.sha256()
        with open(file_path, "rb") as f:
            data = f.read(chunk_size)
            hasher.update(data)
        hasher.update(str(file_path.stat().st_size).encode("utf-8"))
        return hasher.hexdigest()[:16]
    except OSError:
        return ""


class ProcessingJournal:
    """
    Manages persistent SQLite journaling of all document processing operations.
    Thread-safe, handles crashes, reconciles filesystem with database records.
    """

    def __init__(self, db_path: Optional[Union[str, Path]] = None, dry_run: bool = False):
        self.dry_run = dry_run
        self._lock = threading.Lock()

        if db_path:
            self.db_path = Path(db_path).resolve()
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            self._conn = sqlite3.connect(str(self.db_path), check_same_thread=False, timeout=30.0)
        else:
            # In-memory journal if no path provided
            self.db_path = None
            self._conn = sqlite3.connect(":memory:", check_same_thread=False, timeout=30.0)

        self._init_db()

    def _init_db(self) -> None:
        """Initialize tables and indexes with WAL mode for high concurrency."""
        with self._lock:
            cur = self._conn.cursor()
            if self.db_path:
                try:
                    cur.execute("PRAGMA journal_mode=WAL;")
                    cur.execute("PRAGMA synchronous=NORMAL;")
                except sqlite3.DatabaseError:
                    pass

            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS document_journal (
                    source_path TEXT PRIMARY KEY,
                    source_name TEXT NOT NULL,
                    source_size INTEGER NOT NULL,
                    source_mtime REAL NOT NULL,
                    source_hash TEXT,
                    status TEXT NOT NULL,
                    category TEXT,
                    routing_type TEXT,
                    destination_path TEXT,
                    operation TEXT NOT NULL,
                    processed_at TEXT NOT NULL,
                    error TEXT,
                    dry_run INTEGER DEFAULT 0
                );
                """
            )
            cur.execute("CREATE INDEX IF NOT EXISTS idx_journal_status ON document_journal(status);")
            cur.execute("CREATE INDEX IF NOT EXISTS idx_journal_dest ON document_journal(destination_path);")
            self._conn.commit()

    def reconcile_file_status(
        self,
        source_path: Path,
        expected_operation: str = "move",
        is_move_operation: Optional[bool] = None,
    ) -> Optional[Tuple[str, Optional[str], Optional[str]]]:
        """
        Reconciles journal database with the actual filesystem state.

        Returns:
            Tuple (status, category, destination_path) if the file has already been
            safely processed, verified on disk, and can be skipped on resume.
            Returns None if the file needs to be (re-)processed.
        """
        if self.dry_run:
            return None

        if is_move_operation is not None:
            expected_operation = "move" if is_move_operation else "copy"

        src = Path(source_path).resolve()
        src_key = str(src)

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                SELECT source_size, source_mtime, status, category, destination_path, operation
                FROM document_journal
                WHERE source_path = ? AND dry_run = 0
                """,
                (src_key,),
            )
            row = cur.fetchone()

        if not row:
            return None

        rec_size, rec_mtime, rec_status, rec_category, rec_dest_str, rec_op = row

        # If a crash occurred during processing ('IN_PROGRESS'), re-process
        if rec_status == "IN_PROGRESS":
            return None

        dest_path = Path(rec_dest_str) if rec_dest_str else None

        # Case 1: Terminal success / completed routing
        if rec_status in ("COMPLETED", "IDENTIFIED", "NON_KYC", "UNIDENTIFIED"):
            # Check source file: if it still exists, did it change since journal record?
            if src.exists():
                try:
                    cur_st = src.stat()
                    # If size or modification time changed, content modified -> re-process
                    if cur_st.st_size != rec_size or abs(cur_st.st_mtime - rec_mtime) > 1.0:
                        logger.info("File '%s' modified since last run; re-processing.", src.name)
                        return None
                except OSError:
                    return None

                # For 'move' operation: if source file still exists, it is present in
                # the source/input directory and has not yet been moved (or is a new incoming file).
                # In move mode, a processed file must never remain in source. Return None to re-process/move.
                if expected_operation == "move":
                    return None
                else:
                    # For 'copy' operation: both source and destination should exist with valid content
                    if not dest_path or not dest_path.is_file():
                        return None
                    try:
                        if dest_path.stat().st_size == rec_size and rec_size > 0:
                            return (rec_status, rec_category, str(dest_path))
                        else:
                            return None
                    except OSError:
                        return None

            # If source file does NOT exist (already moved successfully):
            # Verify destination exists on disk
            if not dest_path or not dest_path.is_file():
                logger.warning(
                    "Source missing and destination '%s' missing for '%s'. Incomplete operation.",
                    rec_dest_str,
                    src.name,
                )
                return None

            # Verify destination has valid size
            try:
                if dest_path.stat().st_size != rec_size:
                    logger.warning("Destination file size mismatch for '%s'. Re-processing.", src.name)
                    return None
            except OSError:
                return None

            # File is verified on disk!
            return (rec_status, rec_category, str(dest_path))

        return None

    def record_start(
        self,
        source_path: Path,
        operation: str = "move",
    ) -> None:
        """Record that processing for a file has started."""
        if self.dry_run:
            return

        src = Path(source_path).resolve()
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            st = src.stat()
            size = st.st_size
            mtime = st.st_mtime
            h_prefix = compute_quick_hash(src)
        except OSError:
            size = 0
            mtime = 0.0
            h_prefix = ""

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO document_journal (
                    source_path, source_name, source_size, source_mtime, source_hash,
                    status, operation, processed_at, dry_run
                ) VALUES (?, ?, ?, ?, ?, 'IN_PROGRESS', ?, ?, 0)
                ON CONFLICT(source_path) DO UPDATE SET
                    source_size=excluded.source_size,
                    source_mtime=excluded.source_mtime,
                    source_hash=excluded.source_hash,
                    status='IN_PROGRESS',
                    operation=excluded.operation,
                    processed_at=excluded.processed_at,
                    error=NULL,
                    dry_run=0;
                """,
                (str(src), src.name, size, mtime, h_prefix, operation, now_ts),
            )
            self._conn.commit()

    def record_completion(
        self,
        source_path: Path,
        destination_path: Path,
        category: Optional[str],
        routing_type: str,
        operation: str = "move",
        error: Optional[str] = None,
    ) -> None:
        """
        Record verified completion of a file operation.
        Guarantees dry-run operations are never committed as real operations.
        """
        if self.dry_run:
            return

        src = Path(source_path).resolve()
        dest = Path(destination_path).resolve()
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")

        status = "UNIDENTIFIED" if (routing_type == "UNIDENTIFIED" or error) else "COMPLETED"

        try:
            # Use destination size if source was already moved
            if dest.is_file():
                size = dest.stat().st_size
                mtime = dest.stat().st_mtime
            elif src.is_file():
                size = src.stat().st_size
                mtime = src.stat().st_mtime
            else:
                size = 0
                mtime = 0.0
        except OSError:
            size = 0
            mtime = 0.0

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO document_journal (
                    source_path, source_name, source_size, source_mtime, status,
                    category, routing_type, destination_path, operation, processed_at, error, dry_run
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                ON CONFLICT(source_path) DO UPDATE SET
                    status=excluded.status,
                    category=excluded.category,
                    routing_type=excluded.routing_type,
                    destination_path=excluded.destination_path,
                    operation=excluded.operation,
                    processed_at=excluded.processed_at,
                    error=excluded.error,
                    dry_run=0;
                """,
                (
                    str(src),
                    src.name,
                    size,
                    mtime,
                    status,
                    category or "",
                    routing_type,
                    str(dest),
                    operation,
                    now_ts,
                    error or "",
                ),
            )
            self._conn.commit()

    def record_failure(
        self,
        source_path: Path,
        error: str,
        destination_path: Optional[Path] = None,
        operation: str = "move",
    ) -> None:
        """Record an unrecoverable failure or technical error for a document."""
        if self.dry_run:
            return

        src = Path(source_path).resolve()
        dest_str = str(Path(destination_path).resolve()) if destination_path else None
        now_ts = time.strftime("%Y-%m-%d %H:%M:%S")

        try:
            st = src.stat()
            size = st.st_size
            mtime = st.st_mtime
        except OSError:
            size = 0
            mtime = 0.0

        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                """
                INSERT INTO document_journal (
                    source_path, source_name, source_size, source_mtime, status,
                    destination_path, operation, processed_at, error, dry_run
                ) VALUES (?, ?, ?, ?, 'FAILED', ?, ?, ?, ?, 0)
                ON CONFLICT(source_path) DO UPDATE SET
                    status='FAILED',
                    destination_path=excluded.destination_path,
                    operation=excluded.operation,
                    processed_at=excluded.processed_at,
                    error=excluded.error,
                    dry_run=0;
                """,
                (str(src), src.name, size, mtime, dest_str, operation, now_ts, error),
            )
            self._conn.commit()

    def get_file_status(self, source_path: Path) -> Optional[Dict[str, Any]]:
        """Retrieve latest journal state for a specific source path."""
        src = Path(source_path).resolve()
        with self._lock:
            cur = self._conn.cursor()
            cur.execute(
                "SELECT source_path, status, category, routing_type, destination_path, operation, error "
                "FROM document_journal WHERE source_path = ?",
                (str(src),),
            )
            row = cur.fetchone()
            if not row:
                return None
            return {
                "source_path": row[0],
                "status": row[1],
                "category": row[2],
                "routing_type": row[3],
                "destination_path": row[4],
                "operation": row[5],
                "error": row[6],
            }

    def record_completed(
        self,
        source_path: Path,
        category: Optional[str],
        destination_path: Path,
        confidence: float = 1.0,
        operation: str = "move",
        error: Optional[str] = None,
    ) -> None:
        """Alias for record_completion."""
        r_type = "KYC" if category and category not in ("UNKNOWN", "NON_KYC") else ("NON_KYC" if category == "NON_KYC" else "UNIDENTIFIED")
        self.record_completion(
            source_path=source_path,
            destination_path=destination_path,
            category=category,
            routing_type=r_type,
            operation=operation,
            error=error,
        )

    def record_failed(
        self,
        source_path: Path,
        error: str,
        destination_path: Optional[Path] = None,
        operation: str = "move",
    ) -> None:
        """Alias for record_failure."""
        self.record_failure(
            source_path=source_path,
            error=error,
            destination_path=destination_path,
            operation=operation,
        )

    def get_summary_counts(self) -> Dict[str, int]:
        """Return counts of processed documents grouped by status."""
        with self._lock:
            cur = self._conn.cursor()
            cur.execute("SELECT status, COUNT(*) FROM document_journal WHERE dry_run = 0 GROUP BY status;")
            rows = cur.fetchall()
        return {status: count for status, count in rows}

    def close(self) -> None:
        """Safely close database connection."""
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.DatabaseError:
                pass
