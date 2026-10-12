"""
Comprehensive Test Suite for VLM Document Classification & Routing Engine.

Tests all 17 required scenarios and edge cases in an isolated test environment (tmp_path):
1. AADHAAR CARD -> KYC\\test_AADHAAR_CARD.jpg
2. PAN CARD -> KYC\\test_PAN_CARD.jpg
3. VOTER ID -> KYC\\test_VOTER_ID.jpg
4. DRIVING LICENCE -> KYC\\test_DRIVING_LICENCE.jpg
5. RATION CARD -> KYC\\test_RATION_CARD.jpg
6. REGISTRATION CERTIFICATE -> KYC\\test_REGISTRATION_CERTIFICATE.jpg
7. MUNICIPAL TAX RECEIPT -> KYC\\test_MUNICIPAL_RECEIPT.jpg (NOT _MUNICIPAL_TAX_RECEIPT)
8. IT RETURNS -> KYC\\test_IT_RETURNS.jpg
9. NON_KYC -> NON KYC\\test.jpg (No suffix)
10. VLM failure -> Unidentified_files\\test.jpg (Original filename)
11. Empty VLM response -> Unidentified_files\\test.jpg
12. Invalid VLM category -> Unidentified_files\\test.jpg
13. Ollama / HTTP error -> Unidentified_files\\test.jpg
14. Destination collision resolution (_1, _2, etc.)
15. Safe physical move (source no longer exists in Input)
16. Failed move preservation (source remains in Input on failure)
17. Duplicate processing prevention (processed files not reprocessed)
18. Already-classified filenames (no repeated suffix appending)
19. All remaining KYC categories (BANK STATEMENT, CKYC CONFIRMATION, ELECTRICITY BILL, GAS BILL, PHONE BILL, PASSPORT, GOVT ID, CKYC DECLARATION)
"""

from __future__ import annotations

import io
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
import requests

from kyc_sorter.sorter.vlm_router import (
    CLASSIFICATION_SUFFIXES,
    VALID_CATEGORIES,
    BatchSummary,
    CpuMonitorThread,
    classify_document,
    get_destination_folder,
    get_target_filename,
    log_routing_result,
    normalize_category,
    prepare_image_payload,
    process_vlm_batch,
    process_vlm_file,
    resolve_collision,
    route_document,
    safe_move_file,
)


@pytest.fixture
def test_dirs(tmp_path):
    """Creates isolated input and destination directories for testing."""
    input_dir = tmp_path / "Source Files" / "Input"
    input_dir.mkdir(parents=True, exist_ok=True)
    dest_root = tmp_path / "Destination files"
    dest_root.mkdir(parents=True, exist_ok=True)

    kyc_dir = dest_root / "KYC"
    non_kyc_dir = dest_root / "NON KYC"
    unidentified_dir = dest_root / "Unidentified_files"

    for d in (kyc_dir, non_kyc_dir, unidentified_dir):
        d.mkdir(parents=True, exist_ok=True)

    return {
        "input": input_dir,
        "dest_root": dest_root,
        "kyc": kyc_dir,
        "non_kyc": non_kyc_dir,
        "unidentified": unidentified_dir,
    }


def _create_sample_file(directory: Path, filename: str, content: bytes = b"DUMMY_IMAGE_BYTES") -> Path:
    p = directory / filename
    p.write_bytes(content)
    return p


# ============================================================
# 1. AADHAAR CARD
# ============================================================
def test_01_aadhaar_card_routing(test_dirs):
    src = _create_sample_file(test_dirs["input"], "test.jpg")
    res = route_document(src, category="AADHAAR CARD", destination_root=test_dirs["dest_root"])

    assert res.status == "SUCCESS"
    assert res.action == "MOVED"
    assert res.final_filename == "test_AADHAAR_CARD.jpg"
    assert res.destination_path == test_dirs["kyc"] / "test_AADHAAR_CARD.jpg"
    assert res.destination_path.is_file()
    assert not src.exists()


# ============================================================
# 2. PAN CARD
# ============================================================
def test_02_pan_card_routing(test_dirs):
    src = _create_sample_file(test_dirs["input"], "test.jpg")
    res = route_document(src, category="PAN CARD", destination_root=test_dirs["dest_root"])

    assert res.status == "SUCCESS"
    assert res.action == "MOVED"
    assert res.final_filename == "test_PAN_CARD.jpg"
    assert res.destination_path == test_dirs["kyc"] / "test_PAN_CARD.jpg"
    assert res.destination_path.is_file()
    assert not src.exists()


# ============================================================
# 3. VOTER ID
# ============================================================
def test_03_voter_id_routing(test_dirs):
    src = _create_sample_file(test_dirs["input"], "test.jpg")
    res = route_document(src, category="VOTER ID", destination_root=test_dirs["dest_root"])

    assert res.status == "SUCCESS"
    assert res.final_filename == "test_VOTER_ID.jpg"
    assert res.destination_path == test_dirs["kyc"] / "test_VOTER_ID.jpg"
    assert res.destination_path.is_file()
    assert not src.exists()


# ============================================================
# 4. DRIVING LICENCE
# ============================================================
def test_04_driving_licence_routing(test_dirs):
    src = _create_sample_file(test_dirs["input"], "test.jpg")
    res = route_document(src, category="DRIVING LICENCE", destination_root=test_dirs["dest_root"])

    assert res.status == "SUCCESS"
    assert res.final_filename == "test_DRIVING_LICENCE.jpg"
    assert res.destination_path == test_dirs["kyc"] / "test_DRIVING_LICENCE.jpg"
    assert res.destination_path.is_file()
    assert not src.exists()


# ============================================================
# 5. RATION CARD
# ============================================================
def test_05_ration_card_routing(test_dirs):
    src = _create_sample_file(test_dirs["input"], "test.jpg")
    res = route_document(src, category="RATION CARD", destination_root=test_dirs["dest_root"])

    assert res.status == "SUCCESS"
    assert res.final_filename == "test_RATION_CARD.jpg"
    assert res.destination_path == test_dirs["kyc"] / "test_RATION_CARD.jpg"
    assert res.destination_path.is_file()
    assert not src.exists()


# ============================================================
# 6. REGISTRATION CERTIFICATE
# ============================================================
def test_06_registration_certificate_routing(test_dirs):
    src = _create_sample_file(test_dirs["input"], "test.jpg")
    res = route_document(src, category="REGISTRATION CERTIFICATE", destination_root=test_dirs["dest_root"])

    assert res.status == "SUCCESS"
    assert res.final_filename == "test_REGISTRATION_CERTIFICATE.jpg"
    assert res.destination_path == test_dirs["kyc"] / "test_REGISTRATION_CERTIFICATE.jpg"
    assert res.destination_path.is_file()
    assert not src.exists()


# ============================================================
# 7. MUNICIPAL TAX RECEIPT -> _MUNICIPAL_RECEIPT (CRITICAL)
# ============================================================
def test_07_municipal_tax_receipt_suffix(test_dirs):
    src = _create_sample_file(test_dirs["input"], "test.jpg")
    res = route_document(src, category="MUNICIPAL TAX RECEIPT", destination_root=test_dirs["dest_root"])

    assert res.status == "SUCCESS"
    # Suffix MUST be _MUNICIPAL_RECEIPT, NOT _MUNICIPAL_TAX_RECEIPT
    assert res.final_filename == "test_MUNICIPAL_RECEIPT.jpg"
    assert res.final_filename != "test_MUNICIPAL_TAX_RECEIPT.jpg"
    assert res.destination_path == test_dirs["kyc"] / "test_MUNICIPAL_RECEIPT.jpg"
    assert res.destination_path.is_file()
    assert not src.exists()


# ============================================================
# 8. IT RETURNS
# ============================================================
def test_08_it_returns_routing(test_dirs):
    src = _create_sample_file(test_dirs["input"], "test.jpg")
    res = route_document(src, category="IT RETURNS", destination_root=test_dirs["dest_root"])

    assert res.status == "SUCCESS"
    assert res.final_filename == "test_IT_RETURNS.jpg"
    assert res.destination_path == test_dirs["kyc"] / "test_IT_RETURNS.jpg"
    assert res.destination_path.is_file()
    assert not src.exists()


# ============================================================
# 9. NON_KYC (Original filename, Destination files\NON KYC)
# ============================================================
def test_09_non_kyc_routing(test_dirs):
    src = _create_sample_file(test_dirs["input"], "test.jpg")
    res = route_document(src, category="NON_KYC", destination_root=test_dirs["dest_root"])

    assert res.status == "SUCCESS"
    assert res.action == "MOVED"
    assert res.final_filename == "test.jpg"  # NO SUFFIX
    assert res.destination_path == test_dirs["non_kyc"] / "test.jpg"
    assert res.destination_path.is_file()
    assert not src.exists()


# ============================================================
# 10. VLM Failure -> Unidentified_files\test.jpg
# ============================================================
def test_10_vlm_failure_routing(test_dirs):
    src = _create_sample_file(test_dirs["input"], "test.jpg")
    res = route_document(src, category=None, is_error=True, destination_root=test_dirs["dest_root"])

    assert res.status == "UNIDENTIFIED"
    assert res.action == "MOVED"
    assert res.final_filename == "test.jpg"  # Original filename
    assert res.destination_path == test_dirs["unidentified"] / "test.jpg"
    assert res.destination_path.is_file()
    assert not src.exists()


# ============================================================
# 11. Empty VLM Response -> Unidentified_files\test.jpg
# ============================================================
def test_11_empty_vlm_response(test_dirs):
    src = _create_sample_file(test_dirs["input"], "test.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": ""}}

    with patch("requests.post", return_value=mock_resp):
        category, err, _ = classify_document(src, prompt="TEST PROMPT")
        assert category is None
        assert "Empty" in (err or "")

        res = route_document(src, category=category, is_error=True, destination_root=test_dirs["dest_root"])
        assert res.status == "UNIDENTIFIED"
        assert res.final_filename == "test.jpg"
        assert res.destination_path == test_dirs["unidentified"] / "test.jpg"
        assert not src.exists()


# ============================================================
# 12. Invalid VLM Category -> Unidentified_files\test.jpg
# ============================================================
def test_12_invalid_vlm_category(test_dirs):
    src = _create_sample_file(test_dirs["input"], "test.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "Maybe Aadhaar"}}

    with patch("requests.post", return_value=mock_resp):
        category, err, _ = classify_document(src, prompt="TEST PROMPT")
        assert category is None
        assert "Invalid category" in (err or "")

        res = route_document(src, category=category, is_error=True, destination_root=test_dirs["dest_root"])
        assert res.status == "UNIDENTIFIED"
        assert res.final_filename == "test.jpg"
        assert res.destination_path == test_dirs["unidentified"] / "test.jpg"
        assert not src.exists()


# ============================================================
# 13. Ollama / HTTP Error -> Unidentified_files\test.jpg
# ============================================================
def test_13_ollama_http_error(test_dirs):
    src = _create_sample_file(test_dirs["input"], "test.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 500
    mock_resp.text = "Internal Server Error: context size exceeded"

    with patch("requests.post", return_value=mock_resp):
        category, err, _ = classify_document(src, prompt="TEST PROMPT")
        assert category is None
        assert "HTTP error 500" in (err or "")

        res = route_document(src, category=category, is_error=True, destination_root=test_dirs["dest_root"])
        assert res.status == "UNIDENTIFIED"
        assert res.final_filename == "test.jpg"
        assert res.destination_path == test_dirs["unidentified"] / "test.jpg"
        assert not src.exists()


# ============================================================
# 14. Destination Collision Resolution (_1, _2, etc.)
# ============================================================
def test_14_destination_collision_handling(test_dirs):
    # Pre-create KYC\test_AADHAAR_CARD.jpg
    existing1 = test_dirs["kyc"] / "test_AADHAAR_CARD.jpg"
    existing1.write_bytes(b"EXISTING_1")

    # First incoming file with same base name
    src1 = _create_sample_file(test_dirs["input"], "test.jpg", b"FILE_1")
    res1 = route_document(src1, category="AADHAAR CARD", destination_root=test_dirs["dest_root"])

    assert res1.final_filename == "test_AADHAAR_CARD_1.jpg"
    assert res1.destination_path == test_dirs["kyc"] / "test_AADHAAR_CARD_1.jpg"
    assert existing1.read_bytes() == b"EXISTING_1"  # Existing untouched
    assert res1.destination_path.read_bytes() == b"FILE_1"

    # Second incoming file with same base name
    src2 = _create_sample_file(test_dirs["input"], "test.jpg", b"FILE_2")
    res2 = route_document(src2, category="AADHAAR CARD", destination_root=test_dirs["dest_root"])

    assert res2.final_filename == "test_AADHAAR_CARD_2.jpg"
    assert res2.destination_path == test_dirs["kyc"] / "test_AADHAAR_CARD_2.jpg"
    assert existing1.read_bytes() == b"EXISTING_1"
    assert res2.destination_path.read_bytes() == b"FILE_2"


# ============================================================
# 15. Successful Move (Source NO LONGER exists in Input)
# ============================================================
def test_15_successful_move_verifies_source_removed(test_dirs):
    src = _create_sample_file(test_dirs["input"], "photo.jpg", b"ORIGINAL_PHOTO_CONTENT")
    assert src.exists()

    res = route_document(src, category="NON_KYC", destination_root=test_dirs["dest_root"])

    assert res.status == "SUCCESS"
    assert res.action == "MOVED"
    assert not src.exists()
    assert res.destination_path.is_file()
    assert res.destination_path.read_bytes() == b"ORIGINAL_PHOTO_CONTENT"


# ============================================================
# 16. Failed Move Preservation (Source MUST remain)
# ============================================================
def test_16_failed_move_preserves_source(test_dirs):
    src = _create_sample_file(test_dirs["input"], "important_doc.jpg", b"DO_NOT_LOSE_ME")

    # Simulate move failure (e.g. write permission error on destination)
    with patch("shutil.move", side_effect=PermissionError("Access is denied")):
        res = route_document(src, category="AADHAAR CARD", destination_root=test_dirs["dest_root"])

        assert res.status == "FAILED"
        assert res.action == "NOT MOVED"
        assert res.source_exists is True
        assert src.exists()  # Source MUST remain in Input
        assert src.read_bytes() == b"DO_NOT_LOSE_ME"


# ============================================================
# 17. Duplicate Processing Prevention
# ============================================================
def test_17_duplicate_processing_prevention(test_dirs):
    src = _create_sample_file(test_dirs["input"], "single_doc.jpg")
    res = route_document(src, category="PAN CARD", destination_root=test_dirs["dest_root"])

    assert res.status == "SUCCESS"
    assert not src.exists()

    # Re-scanning the input folder finds 0 files because single_doc.jpg was physically moved
    remaining_files = list(test_dirs["input"].glob("*.jpg"))
    assert len(remaining_files) == 0


# ============================================================
# 18. Already-Classified Filename (No Repeated Suffix)
# ============================================================
def test_18_already_classified_filename_no_duplicate_suffix(test_dirs):
    src = _create_sample_file(test_dirs["input"], "ABC_AADHAAR_CARD.jpg")
    res = route_document(src, category="AADHAAR CARD", destination_root=test_dirs["dest_root"])

    # Must NOT become ABC_AADHAAR_CARD_AADHAAR_CARD.jpg
    assert res.final_filename == "ABC_AADHAAR_CARD.jpg"
    assert res.destination_path == test_dirs["kyc"] / "ABC_AADHAAR_CARD.jpg"
    assert not src.exists()


# ============================================================
# 19. All Other KYC Categories Naming Verification
# ============================================================
@pytest.mark.parametrize("category,expected_suffix", [
    ("BANK STATEMENT", "_BANK_STATEMENT"),
    ("CKYC CONFIRMATION", "_CKYC_CONFIRMATION"),
    ("ELECTRICITY BILL", "_ELECTRICITY_BILL"),
    ("GAS BILL", "_GAS_BILL"),
    ("PHONE BILL", "_PHONE_BILL"),
    ("PASSPORT", "_PASSPORT"),
    ("GOVT ID", "_GOVT_ID"),
    ("CKYC DECLARATION", "_CKYC_DECLARATION"),
])
def test_19_remaining_kyc_categories(test_dirs, category, expected_suffix):
    src = _create_sample_file(test_dirs["input"], "doc.jpg")
    res = route_document(src, category=category, destination_root=test_dirs["dest_root"])

    assert res.status == "SUCCESS"
    assert res.final_filename == f"doc{expected_suffix}.jpg"
    assert res.destination_path == test_dirs["kyc"] / f"doc{expected_suffix}.jpg"
    assert not src.exists()


# ============================================================
# 20. Logging Format Output Verification
# ============================================================
def test_20_logging_format_verification(capsys, test_dirs):
    src1 = _create_sample_file(test_dirs["input"], "aadhaar.jpg")
    res1 = route_document(src1, category="AADHAAR CARD", destination_root=test_dirs["dest_root"])
    log_routing_result(res1, index=1, total=3)

    out1 = capsys.readouterr().out
    assert "[1/3] aadhaar.jpg" in out1
    assert "Classification : AADHAAR CARD" in out1
    assert "New filename   : aadhaar_AADHAAR_CARD.jpg" in out1
    assert "Action         : MOVED" in out1
    assert "Source exists  : NO" in out1
    assert "Status         : SUCCESS" in out1

    src2 = _create_sample_file(test_dirs["input"], "photo.jpg")
    res2 = route_document(src2, category="NON_KYC", destination_root=test_dirs["dest_root"])
    log_routing_result(res2, index=2, total=3)

    out2 = capsys.readouterr().out
    assert "[2/3] photo.jpg" in out2
    assert "Classification : NON_KYC" in out2
    assert "Filename       : photo.jpg" in out2
    assert "Action         : MOVED" in out2
    assert "Source exists  : NO" in out2
    assert "Status         : SUCCESS" in out2

    src3 = _create_sample_file(test_dirs["input"], "bad.jpg")
    res3 = route_document(src3, category=None, is_error=True, destination_root=test_dirs["dest_root"])
    log_routing_result(res3, index=3, total=3)

    out3 = capsys.readouterr().out
    assert "[3/3] bad.jpg" in out3
    assert "Classification : FAILED" in out3
    assert "Filename       : bad.jpg" in out3
    assert "Action         : MOVED" in out3
    assert "Source exists  : NO" in out3
    assert "Status         : UNIDENTIFIED" in out3


# ============================================================
# 21. Per-File Detailed Timing Log Format Verification
# ============================================================
def test_21_detailed_timing_log_format_verification(capsys, test_dirs):
    src = _create_sample_file(test_dirs["input"], "aadhaar.jpg")
    res = route_document(src, category="AADHAAR CARD", destination_root=test_dirs["dest_root"])
    res.timings = {
        "File read": 0.01,
        "Image preprocessing": 0.02,
        "VLM inference": 6.84,
        "Response parsing": 0.01,
        "Filename generation": 0.00,
        "Destination resolution": 0.00,
        "File move": 0.03,
        "Move verification": 0.01,
    }
    res.elapsed_time = 6.92
    log_routing_result(res, index=1, total=100)

    out = capsys.readouterr().out
    assert "[1/100] aadhaar.jpg" in out
    assert "TIMING" in out
    assert "File read             : 0.01 sec" in out
    assert "Image preprocessing   : 0.02 sec" in out
    assert "VLM inference         : 6.84 sec" in out
    assert "Response parsing      : 0.01 sec" in out
    assert "Filename generation   : 0.00 sec" in out
    assert "Destination resolution: 0.00 sec" in out
    assert "File move             : 0.03 sec" in out
    assert "Move verification     : 0.01 sec" in out
    assert "TOTAL FILE TIME       : 6.92 sec" in out
    assert "RESULT" in out
    assert "Classification        : AADHAAR CARD" in out
    assert "Final filename        : aadhaar_AADHAAR_CARD.jpg" in out
    assert "Action                : MOVED" in out
    assert "Status                : SUCCESS" in out


# ============================================================
# 22. End-to-End process_vlm_file Timing Instrumentation
# ============================================================
def test_22_process_vlm_file_stage_timings(capsys, test_dirs):
    from kyc_sorter.sorter.vlm_router import process_vlm_file

    src = _create_sample_file(test_dirs["input"], "kyc_doc.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "PAN CARD"}}

    with patch("requests.post", return_value=mock_resp):
        res = process_vlm_file(
            image_file=src,
            index=1,
            total=1,
            destination_root=test_dirs["dest_root"],
            prompt="TEST PROMPT",
        )

        assert res.status == "SUCCESS"
        assert res.action == "MOVED"
        assert res.final_filename == "kyc_doc_PAN_CARD.jpg"
        assert res.destination_path.is_file()
        assert not src.exists()
        assert res.elapsed_time > 0.0

        # Verify all 8 timing stages are present and recorded
        required_stages = [
            "File read",
            "Image preprocessing",
            "VLM inference",
            "Response parsing",
            "Filename generation",
            "Destination resolution",
            "File move",
            "Move verification",
        ]
        for stage in required_stages:
            assert stage in res.timings
            assert isinstance(res.timings[stage], float)
            assert res.timings[stage] >= 0.0

        out = capsys.readouterr().out
        assert "TIMING" in out
        assert "TOTAL FILE TIME" in out
        assert "RESULT" in out
        assert "Classification        : PAN CARD" in out
        assert "Final filename        : kyc_doc_PAN_CARD.jpg" in out


# ============================================================
# 23. Loan Agreement / Sanction Letter False-Positive Prevention
# ============================================================
def test_23_loan_agreement_false_positive_routes_to_non_kyc(test_dirs):
    """
    Verifies that a loan agreement / sanction letter containing CKYC in the filename
    (e.g. RAKESH_20_CKYC_DECLARATION.jpg) and classified as NON_KYC routes to NON KYC
    and NOT KYC, keeping its original filename without a CKYC suffix.
    """
    src = _create_sample_file(test_dirs["input"], "RAKESH_20_CKYC_DECLARATION.jpg")
    res = route_document(src, category="NON_KYC", destination_root=test_dirs["dest_root"])

    assert res.status == "SUCCESS"
    assert res.action == "MOVED"
    assert res.routing_type == "NON_KYC"
    assert res.category == "NON_KYC"
    # Preserves original name, does NOT append _CKYC_CONFIRMATION or _CKYC_DECLARATION
    assert res.final_filename == "RAKESH_20_CKYC_DECLARATION.jpg"
    assert res.destination_path == test_dirs["non_kyc"] / "RAKESH_20_CKYC_DECLARATION.jpg"
    assert res.destination_path.is_file()
    assert not (test_dirs["kyc"] / "RAKESH_20_CKYC_DECLARATION.jpg").exists()
    assert not (test_dirs["kyc"] / "RAKESH_20_CKYC_DECLARATION_CKYC_DECLARATION.jpg").exists()
    assert not src.exists()


# ============================================================
# 24. Genuine CKYC Confirmation Routing
# ============================================================
def test_24_genuine_ckyc_confirmation_routes_to_kyc(test_dirs):
    """
    Verifies that a genuine CKYC confirmation document routes to KYC
    with the official _CKYC_CONFIRMATION suffix.
    """
    src = _create_sample_file(test_dirs["input"], "genuine_ckyc.jpg")
    res = route_document(src, category="CKYC CONFIRMATION", destination_root=test_dirs["dest_root"])

    assert res.status == "SUCCESS"
    assert res.action == "MOVED"
    assert res.routing_type == "KYC"
    assert res.final_filename == "genuine_ckyc_CKYC_CONFIRMATION.jpg"
    assert res.destination_path == test_dirs["kyc"] / "genuine_ckyc_CKYC_CONFIRMATION.jpg"
    assert res.destination_path.is_file()
    assert not src.exists()


# ============================================================
# 25. Batch Processing — Single Worker
# ============================================================
def test_25_process_vlm_batch_single_worker(test_dirs):
    src1 = _create_sample_file(test_dirs["input"], "doc1.jpg")
    src2 = _create_sample_file(test_dirs["input"], "doc2.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "AADHAAR CARD"}}

    with patch("requests.post", return_value=mock_resp):
        summary = process_vlm_batch(
            files=[src1, src2],
            max_workers=1,
            destination_root=test_dirs["dest_root"],
            prompt="TEST PROMPT",
        )

        assert summary.total_files == 2
        assert summary.completed_files == 2
        assert summary.unidentified_files == 0
        assert summary.failed_files == 0
        assert summary.total_duration > 0.0
        assert summary.throughput_fpm > 0.0
        assert not summary.cpu_limit_exceeded


# ============================================================
# 26. Batch Processing — Multi-Worker (2 Workers)
# ============================================================
def test_26_process_vlm_batch_multi_worker(test_dirs):
    src1 = _create_sample_file(test_dirs["input"], "multi1.jpg")
    src2 = _create_sample_file(test_dirs["input"], "multi2.jpg")
    src3 = _create_sample_file(test_dirs["input"], "multi3.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "PAN CARD"}}

    with patch("requests.post", return_value=mock_resp):
        summary = process_vlm_batch(
            files=[src1, src2, src3],
            max_workers=2,
            destination_root=test_dirs["dest_root"],
            prompt="TEST PROMPT",
        )

        assert summary.total_files == 3
        assert summary.completed_files == 3
        assert summary.unidentified_files == 0
        assert summary.failed_files == 0
        assert not summary.cpu_limit_exceeded
        assert len(summary.results) == 3


# ============================================================
# 27. CPU Monitor Thread Sampling & Throttling
# ============================================================
def test_27_cpu_monitor_thread_sampling_and_threshold():
    import time
    monitor = CpuMonitorThread(sample_interval=0.05, cpu_limit=80.0, throttle_cpu=75.0)
    monitor.start()
    time.sleep(0.15)
    monitor.stop()
    monitor.join(timeout=1.0)

    assert len(monitor.samples) > 0
    assert monitor.peak_cpu >= 0.0
    assert isinstance(monitor.cpu_exceeded, bool)


# ============================================================
# 28. Single-Pass Mode — Standard KYC Classification (1 Call)
# ============================================================
def test_28_single_pass_kyc_classification(test_dirs):
    src = _create_sample_file(test_dirs["input"], "single_pass_test.jpg")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "AADHAAR CARD"}}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = process_vlm_file(
            image_file=src,
            destination_root=test_dirs["dest_root"],
            prompt="TEST PROMPT",
            single_pass=True,
        )

        assert mock_post.call_count == 1
        assert result.category == "AADHAAR CARD"
        assert result.routing_type == "KYC"
        assert not result.is_multiple_document
        assert result.destination_path.exists()
        assert "AADHAAR_CARD" in result.destination_path.name


# ============================================================
# 29. Single-Pass Mode — Multiple Documents Detected -> NON_KYC
# ============================================================
def test_29_single_pass_multiple_documents_detection(test_dirs):
    src = _create_sample_file(test_dirs["input"], "multi_doc_single_pass.jpg")
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "MULTIPLE_DOCUMENTS"}}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        result = process_vlm_file(
            image_file=src,
            destination_root=test_dirs["dest_root"],
            prompt="TEST PROMPT",
            single_pass=True,
        )

        assert mock_post.call_count == 1
        assert result.is_multiple_document is True
        assert result.category == "NON_KYC"
        assert result.routing_type == "NON_KYC"
        assert result.destination_path.exists()
        assert "NON KYC" in str(result.destination_path)


# ============================================================
# 30. Image Preprocessing — Resizing & Compression
# ============================================================
def test_30_image_preprocessing_resizing_and_compression(tmp_path):
    import io
    import base64
    from PIL import Image
    from kyc_sorter.sorter.vlm_router import prepare_image_payload

    # Create a large 2400x1600 test image
    img_path = tmp_path / "large_image.jpg"
    img = Image.new("RGB", (2400, 1600), color=(128, 64, 32))
    img.save(img_path, format="JPEG", quality=95)

    b64_str, t_read, t_pre = prepare_image_payload(img_path, max_dimension=1200, quality=85)
    assert isinstance(b64_str, str)
    assert len(b64_str) > 0
    assert t_read >= 0.0
    assert t_pre >= 0.0

    decoded_bytes = base64.b64decode(b64_str)
    processed_img = Image.open(io.BytesIO(decoded_bytes))
    w, h = processed_img.size
    assert max(w, h) == 1200
    assert w == 1200
    assert h == 800


# ============================================================
# 31. Single-Pass Batch Processing
# ============================================================
def test_31_single_pass_batch_processing(test_dirs):
    src1 = _create_sample_file(test_dirs["input"], "sp_batch1.jpg")
    src2 = _create_sample_file(test_dirs["input"], "sp_batch2.jpg")

    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.json.return_value = {"message": {"content": "PAN CARD"}}

    with patch("requests.post", return_value=mock_resp) as mock_post:
        summary = process_vlm_batch(
            files=[src1, src2],
            max_workers=2,
            destination_root=test_dirs["dest_root"],
            prompt="TEST PROMPT",
            single_pass=True,
        )

        assert summary.total_files == 2
        assert summary.completed_files == 2
        assert mock_post.call_count == 2  # exactly 1 call per file in single_pass


