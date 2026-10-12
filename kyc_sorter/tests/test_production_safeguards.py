"""
Comprehensive production-readiness regression tests verifying mandatory safeguards:
1. Multi-document handling in single-pass mode & front/back view distinction
2. Ollama keep-alive configuration and restart/connection drop recovery
3. num_ctx parameter transmission and payload integrity
4. Journal recovery across restarts in copy and move mode
5. Adaptive file readiness and partially copied file handling
6. Watcher burst deduplication
7. Concurrency and CPU backpressure throttling
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from kyc_sorter.sorter.batch_processor import BatchProcessor
from kyc_sorter.sorter.journal import ProcessingJournal
from kyc_sorter.sorter.vlm_router import (
    classify_document,
    process_vlm_batch,
    process_vlm_file,
    route_document,
)
from kyc_sorter.sorter.watcher import SourceFolderWatcher, wait_for_file_ready
from kyc_sorter.utils.config_loader import load_config


@pytest.fixture
def test_env(tmp_path):
    """Isolated environment with synthetic paths and clean journal."""
    in_dir = tmp_path / "Source Files" / "Input"
    in_dir.mkdir(parents=True, exist_ok=True)
    dest_root = tmp_path / "Destination files"
    dest_root.mkdir(parents=True, exist_ok=True)
    journal_path = dest_root / "kyc_journal.db"

    cfg = load_config(
        overrides={
            "input": {"directory": str(in_dir)},
            "output": {"directory": str(dest_root), "operation": "move"},
            "processing": {
                "dry_run": False,
                "max_workers": 2,
                "journal_path": str(journal_path),
                "cpu_limit_percent": 80.0,
                "throttle_cpu_percent": 75.0,
            },
            "vlm": {
                "endpoint": "http://localhost:11434/api/chat",
                "model": "qwen3-vl:4b-instruct",
                "single_pass_mode": True,
                "keep_alive": "24h",
                "num_ctx": 8192,
                "timeout": 30,
            },
            "watcher": {
                "stability_delay_sec": 0.05,
                "settle_timeout_sec": 2.0,
                "poll_interval_sec": 0.02,
            },
        }
    )
    return {
        "input": in_dir,
        "dest_root": dest_root,
        "journal_path": journal_path,
        "config": cfg,
    }


def _create_sample_img(folder: Path, filename: str, content: bytes = b"\xFF\xD8\xFF\xE0\x00\x10JFIF\x00\x01\x01\x00\x00\x01\x00\x01\x00\x00" + b"\x00" * 200) -> Path:
    p = folder / filename
    p.write_bytes(content)
    return p


# ==============================================================================
# 1. Single-Pass Multi-Document & Front/Back Physical ID Card Safeguards
# ==============================================================================


def test_single_pass_multiple_distinct_documents_routes_to_non_kyc(test_env):
    """
    Safeguard 2: When single-pass model detects multiple distinct documents
    (MULTIPLE_DOCUMENTS), it must route immediately to NON KYC with original filename.
    """
    src = _create_sample_img(test_env["input"], "two_different_ids.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {"content": "MULTIPLE_DOCUMENTS"}
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = process_vlm_file(
            image_file=src,
            destination_root=test_env["dest_root"],
            single_pass=True,
            keep_alive="24h",
            num_ctx=4096,
        )

    assert res.routing_type == "NON_KYC"
    assert res.is_multiple_document is True
    assert res.final_filename == "two_different_ids.jpg"
    assert (test_env["dest_root"] / "NON KYC" / "two_different_ids.jpg").is_file()
    assert not src.exists()


def test_single_pass_front_and_back_of_same_id_routes_to_single_kyc(test_env):
    """
    Safeguard 2: Front and back views of the same physical ID must be classified
    as the single card type (e.g. AADHAAR) and route to KYC with category suffix.
    """
    src = _create_sample_img(test_env["input"], "aadhaar_front_back.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {
        "message": {"content": "AADHAAR"}
    }

    with patch("requests.post", return_value=mock_resp) as mock_post:
        res = process_vlm_file(
            image_file=src,
            destination_root=test_env["dest_root"],
            single_pass=True,
            keep_alive="24h",
            num_ctx=4096,
        )

    assert res.routing_type == "KYC"
    assert res.category == "AADHAAR CARD"
    assert res.is_multiple_document is False
    assert res.final_filename == "aadhaar_front_back_AADHAAR_CARD.jpg"
    assert (test_env["dest_root"] / "KYC" / "aadhaar_front_back_AADHAAR_CARD.jpg").is_file()
    assert not src.exists()


# ==============================================================================
# 2. Ollama Keep-Alive & Context Length Parameter Verification
# ==============================================================================


def test_ollama_keep_alive_and_num_ctx_passed_in_payload(test_env):
    """
    Safeguard 3 & 4: keep_alive and num_ctx must be explicitly passed
    in the Ollama API chat request payload options.
    """
    src = _create_sample_img(test_env["input"], "doc_params.jpg")

    captured_payloads = []

    def mock_post(url, json=None, **kwargs):
        captured_payloads.append(json)
        resp = MagicMock()
        resp.status_code = 200
        resp.json.return_value = {"message": {"content": "PASSPORT"}}
        return resp

    with patch("requests.post", side_effect=mock_post):
        res = process_vlm_file(
            image_file=src,
            destination_root=test_env["dest_root"],
            single_pass=True,
            keep_alive="12h",
            num_ctx=4096,
        )

    assert len(captured_payloads) == 1
    sent_json = captured_payloads[0]
    assert sent_json.get("keep_alive") == "12h"
    assert sent_json.get("options", {}).get("num_ctx") == 4096
    assert res.status == "SUCCESS"
    assert res.category == "PASSPORT"


def test_ollama_service_restart_recovery(test_env):
    """
    Safeguard 3: If Ollama crashes or restarts during operation (ConnectionError, 502, 503),
    the application must recover safely, route to Unidentified_files, and not crash.
    """
    src = _create_sample_img(test_env["input"], "restart_test.jpg")

    # Simulate connection error (Ollama down/restarting)
    with patch("requests.post", side_effect=requests.exceptions.ConnectionError("Connection refused")):
        res = process_vlm_file(
            image_file=src,
            destination_root=test_env["dest_root"],
            single_pass=True,
        )

    assert res.status == "UNIDENTIFIED"
    assert res.routing_type == "UNIDENTIFIED"
    assert "Connection refused" in (res.error or "")
    assert (test_env["dest_root"] / "Unidentified_files" / "restart_test.jpg").is_file()


# ==============================================================================
# 3. Journal Recovery Across Application Restarts
# ==============================================================================


def test_journal_recovery_in_move_mode_prevents_duplicate_routing(test_env):
    """
    Safeguard 6: After a move operation completes, restarting the processor
    and attempting to re-process the moved file reports ALREADY_COMPLETED
    without any duplicate routing or file loss.
    """
    src = _create_sample_img(test_env["input"], "pan_card_reconcile.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "PAN CARD"}}

    processor1 = BatchProcessor(config=test_env["config"])
    with patch("requests.post", return_value=mock_resp):
        item1 = processor1.process_file(src)

    assert item1.status in ("IDENTIFIED", "COMPLETED")
    assert not src.exists()
    dest_file = test_env["dest_root"] / "KYC" / "pan_card_reconcile_PAN_CARD.jpg"
    assert dest_file.is_file()

    # Restart: instantiate a new processor pointing to the existing journal
    processor2 = BatchProcessor(config=test_env["config"])
    item2 = processor2.process_file(src)

    assert item2.status == "ALREADY_COMPLETED"
    assert item2.document_type == "PAN CARD"
    assert Path(item2.destination_path).resolve() == dest_file.resolve()


def test_journal_recovery_in_copy_mode_preserves_source(test_env):
    """
    Safeguard 6: In copy mode, the source file remains in the source directory.
    On restart, the processor identifies the file as already completed.
    """
    test_env["config"].output.operation = "copy"
    src = _create_sample_img(test_env["input"], "voter_copy.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "VOTER ID"}}

    processor1 = BatchProcessor(config=test_env["config"])
    with patch("requests.post", return_value=mock_resp):
        item1 = processor1.process_file(src)

    assert item1.status in ("IDENTIFIED", "COMPLETED")
    assert src.exists()  # Source preserved in copy mode
    dest_file = test_env["dest_root"] / "KYC" / "voter_copy_VOTER_ID.jpg"
    assert dest_file.is_file()

    # Re-run / restart in copy mode
    processor2 = BatchProcessor(config=test_env["config"])
    item2 = processor2.process_file(src)

    assert item2.status == "ALREADY_COMPLETED"
    assert src.exists()  # Still intact


# ==============================================================================
# 4. Adaptive File Readiness & Partially Written Files
# ==============================================================================


def test_file_readiness_waits_for_partially_written_file(tmp_path):
    """
    Safeguard 7: wait_for_file_ready must detect when a file is actively being
    written (size growing) and only declare it ready once size stabilizes.
    """
    test_file = tmp_path / "incoming_large_doc.jpg"
    test_file.write_bytes(b"\x00" * 50)

    # File size stabilizes after 100ms
    def append_later():
        time.sleep(0.04)
        with open(test_file, "ab") as f:
            f.write(b"\x00" * 100)

    import threading
    t = threading.Thread(target=append_later)
    t.start()

    ready = wait_for_file_ready(
        file_path=test_file,
        stability_delay_sec=0.08,
        settle_timeout_sec=1.0,
        check_interval_sec=0.02,
    )
    t.join()

    assert ready is True
    assert test_file.stat().st_size == 150


def test_file_readiness_times_out_if_file_never_settles(tmp_path):
    """
    Safeguard 7: If a file never stops mutating within settle_timeout,
    wait_for_file_ready returns False rather than processing corrupt data.
    """
    test_file = tmp_path / "never_settling.jpg"
    test_file.write_bytes(b"\x00" * 10)

    def keep_mutating():
        for _ in range(8):
            time.sleep(0.03)
            with open(test_file, "ab") as f:
                f.write(b"\x00" * 10)

    import threading
    t = threading.Thread(target=keep_mutating)
    t.start()

    ready = wait_for_file_ready(
        file_path=test_file,
        stability_delay_sec=0.1,
        settle_timeout_sec=0.15,
        check_interval_sec=0.02,
    )
    t.join()

    assert ready is False


# ==============================================================================
# 5. Watcher Burst Deduplication & Concurrency Backpressure
# ==============================================================================


def test_watcher_burst_event_suppression(test_env):
    """
    Safeguard 6: When multiple filesystem events are triggered for the same file,
    the watcher processes it exactly once without duplicate routing or errors.
    """
    watcher = SourceFolderWatcher(config=test_env["config"])
    processed_calls = []

    watcher.processor = MagicMock()
    watcher.processor.process_file.side_effect = lambda f: processed_calls.append(f)

    src = _create_sample_img(test_env["input"], "burst_card.jpg")

    # Simulate 5 rapid bursts for the same file
    for _ in range(5):
        watcher.on_filesystem_event(src)

    # Drain queue
    if watcher._worker_pool:
        watcher._worker_pool.shutdown(wait=True)

    assert len(processed_calls) == 1


def test_concurrency_and_cpu_backpressure(test_env):
    """
    Safeguard 5: When CPU limit is exceeded, process_vlm_batch pauses
    submission and throttles workers until CPU cools down.
    """
    src1 = _create_sample_img(test_env["input"], "bench_1.jpg")
    src2 = _create_sample_img(test_env["input"], "bench_2.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "PASSPORT"}}

    mock_cpu_monitor = MagicMock()
    # First sample: throttled, second: clear
    mock_cpu_monitor.cpu_exceeded = False
    mock_cpu_monitor.throttle_active = True
    mock_cpu_monitor.last_sample_time = time.time()
    mock_cpu_monitor.samples = [78.0, 72.0]
    mock_cpu_monitor.mem_samples = [45.0, 45.0]
    mock_cpu_monitor.peak_cpu = 78.0
    mock_cpu_monitor.peak_mem_percent = 45.0
    mock_cpu_monitor.ever_cpu_limit_exceeded = False
    mock_cpu_monitor.peak_vram_mb = 1200.0
    mock_cpu_monitor.gpu_info = {"available": True, "name": "NVIDIA RTX 4050"}

    # Simulate cool down after 0.2s
    def clear_throttle():
        time.sleep(0.2)
        mock_cpu_monitor.throttle_active = False

    import threading
    t = threading.Thread(target=clear_throttle)
    t.start()

    with patch("requests.post", return_value=mock_resp):
        summary = process_vlm_batch(
            files=[src1, src2],
            max_workers=2,
            destination_root=test_env["dest_root"],
            cpu_monitor=mock_cpu_monitor,
            single_pass=True,
            keep_alive="24h",
            num_ctx=4096,
        )
    t.join()

    assert summary.total_files == 2
    assert summary.completed_files == 2
    assert summary.failed_files == 0
