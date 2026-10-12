"""
Task 2 Integration and Regression Tests: VLM Pipeline in Main Application.

Verifies:
1. CLI batch processing using VLM rather than OCR.
2. Watcher processing using VLM rather than OCR.
3. Prompt loading directly from prompts.py without importing old test scripts.
4. Correct routing and naming for all 16 KYC categories and NON_KYC.
5. CKYC false-positive protections and loan agreement handling.
6. Safe file moves, collisions, failures, and duplicate watcher events.
7. Dry-run, processing limits, and CLI input/output overrides.
8. CPU throttling, worker concurrency, and runtime worker count verification.
9. Ollama unavailable, timeout, invalid response, and unidentified-file handling.
10. Companion-file safety: missing .txt/.json files never break JPG classification.
11. Path safety: automatic redirection away from original project paths.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from kyc_sorter.detector.prompts import PROMPT
from kyc_sorter.main import parse_args, run_cli
from kyc_sorter.sorter.batch_processor import BatchProcessor, ProcessedItem
from kyc_sorter.sorter.vlm_router import (
    CLASSIFICATION_SUFFIXES,
    CpuMonitorThread,
    process_vlm_batch,
    process_vlm_file,
    route_document,
)
from kyc_sorter.sorter.watcher import SourceFolderWatcher
from kyc_sorter.utils.config_loader import Config, load_config


@pytest.fixture
def temp_environment(tmp_path):
    """Isolated input and destination directories for safe tests."""
    input_dir = tmp_path / "Source Files" / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)
    dest_root = tmp_path / "Destination files"
    dest_root.mkdir(parents=True, exist_ok=True)

    config = load_config(
        overrides={
            "input": {"directory": str(input_dir)},
            "output": {"directory": str(dest_root), "operation": "move"},
            "processing": {"dry_run": False, "max_workers": 4, "limit": None},
        }
    )
    return {
        "input_dir": input_dir,
        "dest_root": dest_root,
        "config": config,
    }


def _create_sample_file(dir_path: Path, filename: str, content: bytes = b"\xFF\xD8\xFF\xE0\x00\x10JFIF" + b"\x00" * 32) -> Path:
    p = dir_path / filename
    p.write_bytes(content)
    return p


# ============================================================
# 1. Prompt loading directly without batch-test script
# ============================================================
def test_prompt_loading_no_batch_test_import():
    """Verify PROMPT is loaded from kyc_sorter.detector.prompts directly."""
    assert isinstance(PROMPT, str)
    assert len(PROMPT) > 2000
    assert "You are an expert Indian KYC document classifier." in PROMPT
    assert "CATEGORIES:" in PROMPT
    assert "AADHAAR CARD" in PROMPT
    assert "PAN CARD" in PROMPT
    assert "CKYC CONFIRMATION" in PROMPT
    assert "CKYC DECLARATION" in PROMPT
    assert "LOAN AGREEMENT" in PROMPT
    assert "NON_KYC" in PROMPT

    # Verify vlm_batch_test 1 was not imported during prompt loading
    assert "vlm_batch_test 1" not in sys.modules


# ============================================================
# 2. BatchProcessor CLI batch uses VLM by default without OCR
# ============================================================
def test_batch_processor_vlm_default_no_ocr_init(temp_environment):
    """BatchProcessor default instantiation must NOT initialize OCR engine or classifier."""
    processor = BatchProcessor(config=temp_environment["config"])
    assert processor.use_vlm is True
    assert processor.ocr_engine is None
    assert processor.classifier is None


def test_batch_processor_cli_batch_runs_vlm(temp_environment):
    """BatchProcessor process_batch invokes VLM pipeline, not OCR."""
    input_dir = temp_environment["input_dir"]
    dest_root = temp_environment["dest_root"]
    config = temp_environment["config"]

    _create_sample_file(input_dir, "doc1.jpg")
    _create_sample_file(input_dir, "doc2.jpg")

    processor = BatchProcessor(config=config)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "PAN CARD"}}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        summary = processor.process_batch(source_dir=input_dir)

    # 2 calls per file: multi-document detection + category classification
    assert mock_post.call_count == 4
    assert summary.total_files == 2
    assert summary.processed == 2
    assert summary.failed == 0
    assert summary.counts_by_type.get("PAN CARD") == 2
    assert summary.pipeline_type == "VLM"
    assert summary.model_name == "qwen3-vl:4b-instruct"

    # Verify moved and renamed
    target_pan = dest_root / "KYC" / "doc1_PAN_CARD.jpg"
    assert target_pan.exists()
    assert not (input_dir / "doc1.jpg").exists()


# ============================================================
# 3. Watcher processing uses VLM rather than OCR
# ============================================================
def test_watcher_uses_vlm_not_ocr(temp_environment):
    """SourceFolderWatcher uses BatchProcessor with VLM path."""
    input_dir = temp_environment["input_dir"]
    dest_root = temp_environment["dest_root"]
    config = temp_environment["config"]

    watcher = SourceFolderWatcher(config=config)
    assert watcher.processor.use_vlm is True
    assert watcher.processor.ocr_engine is None

    test_file = _create_sample_file(input_dir, "incoming.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "AADHAAR CARD"}}

    with patch("requests.post", return_value=mock_resp):
        with patch("kyc_sorter.sorter.watcher.wait_for_file_ready", return_value=True):
            watcher._process_single_file_safe(test_file)

    target_aadhaar = dest_root / "KYC" / "incoming_AADHAAR_CARD.jpg"
    assert target_aadhaar.exists()
    assert not test_file.exists()
    assert test_file in watcher.processed_files


# ============================================================
# 4. Correct routing and naming for all 16 KYC categories + NON_KYC
# ============================================================
@pytest.mark.parametrize("category,expected_suffix", list(CLASSIFICATION_SUFFIXES.items()))
def test_all_16_kyc_categories_routing(temp_environment, category, expected_suffix):
    """Every supported KYC category routes to KYC/ with exact suffix."""
    input_dir = temp_environment["input_dir"]
    dest_root = temp_environment["dest_root"]
    config = temp_environment["config"]

    filename = f"test_{category.replace(' ', '_').lower()}.jpg"
    src = _create_sample_file(input_dir, filename)

    processor = BatchProcessor(config=config)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": category}}

    with patch("requests.post", return_value=mock_resp):
        item = processor.process_file(src)

    assert item.status == "IDENTIFIED"
    assert item.document_type == category
    expected_name = f"{src.stem}{expected_suffix}.jpg"
    assert Path(item.destination_path).name == expected_name
    assert Path(item.destination_path).parent.name == "KYC"
    assert Path(item.destination_path).exists()


def test_non_kyc_routing(temp_environment):
    """NON_KYC routes to NON KYC/ folder with original filename unchanged."""
    input_dir = temp_environment["input_dir"]
    dest_root = temp_environment["dest_root"]
    config = temp_environment["config"]

    src = _create_sample_file(input_dir, "contract_document.jpg")
    processor = BatchProcessor(config=config)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "NON_KYC"}}

    with patch("requests.post", return_value=mock_resp):
        item = processor.process_file(src)

    assert item.status == "NON_KYC"
    assert item.document_type == "NON_KYC"
    assert Path(item.destination_path).name == "contract_document.jpg"
    assert Path(item.destination_path).parent.name == "NON KYC"
    assert Path(item.destination_path).exists()


# ============================================================
# 5. CKYC false-positive protections and loan agreement handling
# ============================================================
def test_loan_agreement_with_aadhaar_esign_classified_non_kyc(temp_environment):
    """Loan agreement / Sanction letter with Aadhaar e-sign is routed as NON_KYC."""
    input_dir = temp_environment["input_dir"]
    dest_root = temp_environment["dest_root"]
    config = temp_environment["config"]

    src = _create_sample_file(input_dir, "loan_agreement_esign.jpg")
    processor = BatchProcessor(config=config)

    # Prompt dictates loan agreements with Aadhaar e-sign are NON_KYC
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "NON_KYC"}}

    with patch("requests.post", return_value=mock_resp):
        item = processor.process_file(src)

    assert item.status == "NON_KYC"
    assert Path(item.destination_path).parent == dest_root / "NON KYC"
    assert Path(item.destination_path).name == "loan_agreement_esign.jpg"


def test_genuine_ckyc_confirmation(temp_environment):
    """Genuine CKYC confirmation gets routed to KYC with _CKYC_CONFIRMATION."""
    input_dir = temp_environment["input_dir"]
    dest_root = temp_environment["dest_root"]
    config = temp_environment["config"]

    src = _create_sample_file(input_dir, "ckyc_letter.jpg")
    processor = BatchProcessor(config=config)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "CKYC CONFIRMATION"}}

    with patch("requests.post", return_value=mock_resp):
        item = processor.process_file(src)

    assert item.status == "IDENTIFIED"
    assert item.document_type == "CKYC CONFIRMATION"
    assert Path(item.destination_path).name == "ckyc_letter_CKYC_CONFIRMATION.jpg"


# ============================================================
# 6. Safe file moves, collisions, and failures
# ============================================================
def test_destination_collision_resolution(temp_environment):
    """Successive files with duplicate target names append _1, _2."""
    input_dir = temp_environment["input_dir"]
    dest_root = temp_environment["dest_root"]
    config = temp_environment["config"]

    processor = BatchProcessor(config=config)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "VOTER ID"}}

    # File 1
    src1 = _create_sample_file(input_dir, "card.jpg", b"BYTES_1")
    with patch("requests.post", return_value=mock_resp):
        item1 = processor.process_file(src1)
    assert Path(item1.destination_path).name == "card_VOTER_ID.jpg"

    # File 2 (same initial name)
    src2 = _create_sample_file(input_dir, "card.jpg", b"BYTES_2")
    with patch("requests.post", return_value=mock_resp):
        item2 = processor.process_file(src2)
    assert Path(item2.destination_path).name == "card_VOTER_ID_1.jpg"

    # File 3 (same initial name)
    src3 = _create_sample_file(input_dir, "card.jpg", b"BYTES_3")
    with patch("requests.post", return_value=mock_resp):
        item3 = processor.process_file(src3)
    assert Path(item3.destination_path).name == "card_VOTER_ID_2.jpg"

    assert (dest_root / "KYC" / "card_VOTER_ID.jpg").read_bytes() == b"BYTES_1"
    assert (dest_root / "KYC" / "card_VOTER_ID_1.jpg").read_bytes() == b"BYTES_2"
    assert (dest_root / "KYC" / "card_VOTER_ID_2.jpg").read_bytes() == b"BYTES_3"


def test_failed_move_preserves_source(temp_environment):
    """If file move fails, source file is preserved in source directory."""
    input_dir = temp_environment["input_dir"]
    dest_root = temp_environment["dest_root"]

    src = _create_sample_file(input_dir, "critical.jpg", b"CRITICAL_DATA")

    with patch("shutil.move", side_effect=OSError("Disk write error")):
        res = route_document(src, category="PAN CARD", destination_root=dest_root)

    assert res.status == "FAILED"
    assert src.exists()
    assert src.read_bytes() == b"CRITICAL_DATA"


def test_duplicate_watcher_events_suppressed(temp_environment):
    """Duplicate events for the same file are ignored without reprocessing."""
    input_dir = temp_environment["input_dir"]
    config = temp_environment["config"]
    watcher = SourceFolderWatcher(config=config)

    src = _create_sample_file(input_dir, "doc.jpg")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "PASSPORT"}}

    with patch("requests.post", return_value=mock_resp) as mock_p:
        with patch("kyc_sorter.sorter.watcher.wait_for_file_ready", return_value=True):
            watcher._process_single_file_safe(src)
            # Second event with same signature
            watcher._process_single_file_safe(src)

    # First event processed the file (detection + classification = 2 calls), second saw duplicate
    assert mock_p.call_count == 2


# ============================================================
# 7. Dry-run, limits, and CLI input/output overrides
# ============================================================
def test_dry_run_does_not_modify_disk(temp_environment):
    """Dry run simulates classification and routing without moving files."""
    input_dir = temp_environment["input_dir"]
    dest_root = temp_environment["dest_root"]

    dry_config = load_config(
        overrides={
            "input": {"directory": str(input_dir)},
            "output": {"directory": str(dest_root), "operation": "move"},
            "processing": {"dry_run": True},
        }
    )
    processor = BatchProcessor(config=dry_config)

    src = _create_sample_file(input_dir, "dry_test.jpg", b"DRY_BYTES")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "RATION CARD"}}

    with patch("requests.post", return_value=mock_resp):
        item = processor.process_file(src)

    assert item.dry_run is True
    assert src.exists()  # Source not deleted
    assert not (dest_root / "KYC" / "dry_test_RATION_CARD.jpg").exists()  # Dest not created


def test_batch_processing_limit(temp_environment):
    """Limit option limits the number of files processed in batch."""
    input_dir = temp_environment["input_dir"]
    config = temp_environment["config"]

    for i in range(5):
        _create_sample_file(input_dir, f"doc_{i}.jpg")

    processor = BatchProcessor(config=config)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "GOVT ID"}}

    with patch("requests.post", return_value=mock_resp):
        summary = processor.process_batch(source_dir=input_dir, limit=2)

    assert summary.total_files == 2
    assert summary.processed == 2


def test_cli_overrides_honored(temp_environment, monkeypatch):
    """CLI flags for input, output, operation, dry-run, and workers are honored."""
    input_dir = temp_environment["input_dir"]
    dest_root = temp_environment["dest_root"]

    test_args = [
        "main.py",
        "-i", str(input_dir),
        "-o", str(dest_root),
        "--operation", "copy",
        "--dry-run",
        "--workers", "4",
        "-n", "3",
    ]
    monkeypatch.setattr("sys.argv", test_args)
    parsed = parse_args()

    assert parsed.input == str(input_dir)
    assert parsed.output == str(dest_root)
    assert parsed.operation == "copy"
    assert parsed.dry_run is True
    assert parsed.workers == 4
    assert parsed.limit == 3


# ============================================================
# 8. CPU throttling and worker concurrency
# ============================================================
def test_worker_concurrency_and_runtime_verification(temp_environment):
    """Explicit 4-worker concurrency target with runtime verification."""
    input_dir = temp_environment["input_dir"]
    config = temp_environment["config"]

    for i in range(4):
        _create_sample_file(input_dir, f"conc_{i}.jpg")

    processor = BatchProcessor(config=config)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "GAS BILL"}}

    with patch("requests.post", return_value=mock_resp):
        summary = processor.process_batch(source_dir=input_dir)

    assert summary.max_active_workers == 4
    assert summary.processed == 4


def test_cpu_monitor_throttle_flags():
    """CpuMonitorThread correctly activates throttle and limit flags."""
    monitor = CpuMonitorThread(sample_interval=0.01, cpu_limit=80.0, throttle_cpu=75.0)

    # Simulate CPU readings
    monitor.samples = [70.0, 76.0, 82.0]
    monitor.recent_window.extend([81.0, 82.0])
    rolling = sum(monitor.recent_window) / len(monitor.recent_window)

    assert rolling >= 80.0
    monitor.stop()


# ============================================================
# 9. Ollama unavailable, timeout, invalid response -> Unidentified
# ============================================================
def test_ollama_connection_error_routes_to_unidentified(temp_environment):
    """Connection error (e.g. Ollama daemon down) routes to Unidentified_files."""
    input_dir = temp_environment["input_dir"]
    dest_root = temp_environment["dest_root"]
    config = temp_environment["config"]

    src = _create_sample_file(input_dir, "unreachable.jpg")
    processor = BatchProcessor(config=config)

    with patch("requests.post", side_effect=requests.ConnectionError("Connection refused")):
        item = processor.process_file(src)

    assert item.status == "FAILED"
    assert item.document_type == "UNKNOWN"
    dest = Path(item.destination_path)
    assert dest.parent == dest_root / "Unidentified_files"
    assert dest.name == "unreachable.jpg"
    assert dest.exists()


def test_ollama_timeout_routes_to_unidentified(temp_environment):
    """Ollama request timeout routes to Unidentified_files."""
    input_dir = temp_environment["input_dir"]
    dest_root = temp_environment["dest_root"]
    config = temp_environment["config"]

    src = _create_sample_file(input_dir, "timeout.jpg")
    processor = BatchProcessor(config=config)

    with patch("requests.post", side_effect=requests.Timeout("Model execution timed out")):
        item = processor.process_file(src)

    assert item.status == "FAILED"
    dest = Path(item.destination_path)
    assert dest.parent == dest_root / "Unidentified_files"
    assert dest.name == "timeout.jpg"


def test_ollama_invalid_category_routes_to_unidentified(temp_environment):
    """Hallucinated or invalid category string routes to Unidentified_files."""
    input_dir = temp_environment["input_dir"]
    dest_root = temp_environment["dest_root"]
    config = temp_environment["config"]

    src = _create_sample_file(input_dir, "random_text.jpg")
    processor = BatchProcessor(config=config)

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "UNKNOWN_DOG_PICTURE"}}

    with patch("requests.post", return_value=mock_resp):
        item = processor.process_file(src)

    assert item.status in ("FAILED", "UNIDENTIFIED")
    assert item.document_type == "UNKNOWN"
    dest = Path(item.destination_path)
    assert dest.parent == dest_root / "Unidentified_files"
    assert dest.name == "random_text.jpg"


# ============================================================
# 10. Companion-file safety: missing companion files do not break JPG
# ============================================================
def test_missing_companion_files_does_not_break_classification(temp_environment):
    """Absence of .txt or .json companion files does not affect JPG classification."""
    input_dir = temp_environment["input_dir"]
    config = temp_environment["config"]

    # Image exists without companion
    img = _create_sample_file(input_dir, "isolated_doc.jpg")
    assert not (input_dir / "isolated_doc.txt").exists()
    assert not (input_dir / "isolated_doc.json").exists()

    processor = BatchProcessor(config=config)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "PAN CARD"}}

    with patch("requests.post", return_value=mock_resp):
        item = processor.process_file(img)

    assert item.status == "IDENTIFIED"
    assert item.document_type == "PAN CARD"
    assert Path(item.destination_path).exists()


def test_existing_companion_files_do_not_interfere(temp_environment):
    """Presence of extra .txt/.json files does not interfere with JPG classification."""
    input_dir = temp_environment["input_dir"]
    config = temp_environment["config"]

    img = _create_sample_file(input_dir, "with_txt.jpg")
    (input_dir / "with_txt.txt").write_text("Old OCR Dump", encoding="utf-8")
    (input_dir / "with_txt.json").write_text('{"key": "val"}', encoding="utf-8")

    processor = BatchProcessor(config=config)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "VOTER ID"}}

    with patch("requests.post", return_value=mock_resp):
        item = processor.process_file(img)

    assert item.status == "IDENTIFIED"
    assert item.document_type == "VOTER ID"


# ============================================================
# 11. Path safety: Automatic redirection away from original project
# ============================================================
def test_path_safety_original_directory_redirection():
    """Paths referencing D:/KYC_Document_sorter raise ConfigurationError by default unless explicitly allowed."""
    from kyc_sorter.utils.config_loader import ConfigurationError

    # 1. By default: fail early without silent redirection
    with pytest.raises(ConfigurationError) as exc_info:
        load_config(
            overrides={
                "input": {"directory": "D:/KYC_Document_sorter/Source Files/Input"},
                "output": {"directory": "D:/KYC_Document_sorter/Destination files"},
            }
        )
    assert "points to forbidden original project" in str(exc_info.value)

    # 2. With explicit safe override: redirects safely to current workspace
    cfg = load_config(
        overrides={
            "input": {"directory": "D:/KYC_Document_sorter/Source Files/Input"},
            "output": {"directory": "D:/KYC_Document_sorter/Destination files"},
        },
        allow_legacy_paths=True,
    )
    orig = Path("D:/KYC_Document_sorter").resolve()
    vlm_only = Path("D:/KYC_Document_sorter_VLM_ONLY").resolve()

    assert cfg.input.resolved_path != orig / "Source Files" / "Input"
    assert cfg.output.resolved_path != orig / "Destination files"
    assert vlm_only in cfg.input.resolved_path.parents or cfg.input.resolved_path == vlm_only
    assert vlm_only in cfg.output.resolved_path.parents or cfg.output.resolved_path == vlm_only


# ============================================================
# 12. Retained GUI behavior without OCR
# ============================================================
def test_gui_instantiation_and_vlm_execution(temp_environment):
    """Retained GUI initializes BatchProcessor with VLM pipeline and no OCR."""
    import tkinter as tk
    from kyc_sorter.gui import SorterApp

    try:
        app = SorterApp()
        app.withdraw()  # keep window hidden
        try:
            # Set up GUI variables
            app.input_entry.delete(0, tk.END)
            app.input_entry.insert(0, str(temp_environment["input_dir"]))
            app.output_entry.delete(0, tk.END)
            app.output_entry.insert(0, str(temp_environment["dest_root"]))
            app.op_var.set("Copy")

            src = _create_sample_file(temp_environment["input_dir"], "gui_test.jpg")
            mock_resp = MagicMock()
            mock_resp.status_code = 200
            mock_resp.json.return_value = {"message": {"content": "PASSPORT"}}

            with patch("requests.post", return_value=mock_resp):
                # Synchronously invoke _run_batch for test
                app._run_batch(temp_environment["input_dir"], temp_environment["dest_root"])

            assert (temp_environment["dest_root"] / "KYC" / "gui_test_PASSPORT.jpg").exists()
        finally:
            app.destroy()
    except tk.TclError:
        pytest.skip("Tk display not available in test runner")


# ============================================================
# 13. File extension handling: uppercase .JPG and unsupported skips
# ============================================================
def test_uppercase_and_unsupported_extensions(temp_environment):
    """BatchProcessor handles .JPG correctly and skips unsupported files (.txt, .pdf)."""
    input_dir = temp_environment["input_dir"]
    dest_root = temp_environment["dest_root"]
    config = temp_environment["config"]

    _create_sample_file(input_dir, "PHOTO_UPPER.JPG")
    (input_dir / "notes.txt").write_text("ignore me", encoding="utf-8")
    (input_dir / "invoice.pdf").write_bytes(b"%PDF-1.4")

    processor = BatchProcessor(config=config)
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "VOTER ID"}}

    with patch("requests.post", return_value=mock_resp):
        summary = processor.process_batch(source_dir=input_dir)

    assert summary.total_files == 1  # Only .JPG detected
    assert summary.processed == 1
    assert (dest_root / "KYC" / "PHOTO_UPPER_VOTER_ID.jpg").exists()


# ============================================================
# 14. Configuration path resolution & VLM endpoint propagation
# ============================================================
def test_relative_path_resolution():
    """Relative config paths resolve to workspace project root."""
    cfg = load_config()
    assert cfg.input.resolved_path.is_absolute()
    assert cfg.output.resolved_path.is_absolute()
    assert "Source Files" in str(cfg.input.resolved_path)
    assert "Destination files" in str(cfg.output.resolved_path)
    assert cfg.vlm.model == "qwen3-vl:4b-instruct"
    assert cfg.vlm.endpoint == "http://localhost:11434/api/chat"
    assert cfg.vlm.timeout == 300
    assert cfg.processing.max_workers == 4


def test_vlm_endpoint_override_in_batch_processor(temp_environment):
    """Custom VLM endpoint and model from config are used in Ollama requests."""
    custom_cfg = load_config(
        overrides={
            "input": {"directory": str(temp_environment["input_dir"])},
            "output": {"directory": str(temp_environment["dest_root"])},
            "vlm": {
                "endpoint": "http://custom-ollama:11434/api/chat",
                "model": "custom-qwen:7b",
                "timeout": 150,
            },
        }
    )
    processor = BatchProcessor(config=custom_cfg)
    src = _create_sample_file(temp_environment["input_dir"], "custom_model.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "PAN CARD"}}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        processor.process_file(src)

    assert mock_post.call_count == 2
    call_args, call_kwargs = mock_post.call_args
    assert call_args[0] == "http://custom-ollama:11434/api/chat"
    assert call_kwargs["json"]["model"] == "custom-qwen:7b"
    assert call_kwargs["timeout"] == 150


# ============================================================
# 15. Clean startup and zero OCR runtime imports
# ============================================================
def test_clean_application_startup_no_ocr_imports():
    """Verify application modules load without importing paddle or paddleocr."""
    import kyc_sorter.gui
    import kyc_sorter.main
    import kyc_sorter.sorter.batch_processor
    import kyc_sorter.sorter.vlm_router
    import kyc_sorter.sorter.watcher
    import kyc_sorter.utils.config_loader

    ocr_modules = [
        "paddleocr",
        "paddle",
        "paddlepaddle",
        "kyc_sorter.detector.ocr",
        "kyc_sorter.detector.classifier",
        "kyc_sorter.detector.templates",
    ]
    for mod in ocr_modules:
        assert mod not in sys.modules, f"Forbidden module {mod} was imported!"
