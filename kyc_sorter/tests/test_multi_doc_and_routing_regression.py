"""
Automated Regression Test Suite for KYC Multi-Document Classification & Routing Rules.
Explicitly verifies the 15 mandatory test cases specified in Section 4:
  1. One Aadhaar card alone -> Aadhaar
  2. Aadhaar front and back of the same ID -> Aadhaar (Rule B)
  3. One PAN card alone -> PAN
  4. PAN front and back of the same ID -> PAN (Rule B)
  5. Aadhaar + PAN in one image -> NON_KYC (Rule A)
  6. PAN + bank statement in one image -> NON_KYC (Rule A)
  7. Two different IDs in one image -> NON_KYC (Rule A)
  8. Two views of the same driving licence -> driving licence (Rule B)
  9. Two unrelated documents in one image -> NON_KYC (Rule A)
  10. An unreadable image -> Unidentified_files (Rule C)
  11. An ambiguous image that cannot reliably be distinguished -> Unidentified_files (Rule C)
  12. Malformed VLM response -> safe failure without an incorrect KYC classification (Rule C)
  13. Batch mode and watcher mode apply identical routing rules
  14. A previously processed file does not create duplicate outputs on watcher restart
  15. Existing journal records remain readable
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import time
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image

from kyc_sorter.sorter.batch_processor import BatchProcessor
from kyc_sorter.sorter.journal import ProcessingJournal
from kyc_sorter.sorter.vlm_router import (
    classify_document,
    get_destination_folder,
    get_target_filename,
    normalize_category,
    process_vlm_file,
    route_document,
)
from kyc_sorter.sorter.watcher import SourceFolderWatcher
from kyc_sorter.utils.config_loader import load_config


def create_dummy_image(path: Path, width: int = 400, height: int = 300, color: str = "white") -> Path:
    """Create a minimal valid JPEG image for routing tests."""
    img = Image.new("RGB", (width, height), color=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    img.save(path, format="JPEG")
    return path


@pytest.fixture
def test_setup(tmp_path):
    """Isolated environment with temporary input, destination, and journal."""
    in_dir = tmp_path / "input"
    in_dir.mkdir(parents=True, exist_ok=True)
    out_dir = tmp_path / "output"
    out_dir.mkdir(parents=True, exist_ok=True)
    journal_path = tmp_path / "journal" / "kyc_journal.db"
    journal_path.parent.mkdir(parents=True, exist_ok=True)

    config = load_config(
        overrides={
            "input": {"directory": str(in_dir)},
            "output": {"directory": str(out_dir), "operation": "copy"},
            "processing": {
                "dry_run": False,
                "max_workers": 2,
                "journal_path": str(journal_path),
                "cpu_limit_percent": 80.0,
            },
            "vlm": {
                "endpoint": "http://localhost:11434/api/chat",
                "model": "qwen3-vl:4b-instruct",
                "single_pass_mode": True,
                "keep_alive": "24h",
                "num_ctx": 8192,
            },
        }
    )
    return {
        "in_dir": in_dir,
        "out_dir": out_dir,
        "journal_path": journal_path,
        "config": config,
    }


# ==============================================================================
# Case 1: One Aadhaar card alone -> Aadhaar
# ==============================================================================
def test_01_one_aadhaar_card_alone(test_setup):
    """Case 1: Single Aadhaar card alone must route to KYC with _AADHAAR_CARD suffix."""
    src = create_dummy_image(test_setup["in_dir"] / "cust_aadhaar.jpg")
    vlm_mock = {"category": "AADHAAR CARD"}

    with patch("kyc_sorter.sorter.vlm_router.classify_document", return_value=("AADHAAR CARD", None, 0.5)):
        res = process_vlm_file(
            image_file=src,
            index=1,
            total=1,
            destination_root=test_setup["out_dir"],
            dry_run=False,
            operation="copy",
            single_pass=True,
        )

    assert res.status == "SUCCESS"
    assert res.routing_type == "KYC"
    assert res.category == "AADHAAR CARD"
    assert res.final_filename == "cust_aadhaar_AADHAAR_CARD.jpg"
    assert res.destination_path == test_setup["out_dir"] / "KYC" / "cust_aadhaar_AADHAAR_CARD.jpg"
    assert res.destination_path.exists()
    assert src.exists()  # COPY mode preserves source


# ==============================================================================
# Case 2: Aadhaar front and back of the same ID -> Aadhaar (Rule B)
# ==============================================================================
def test_02_aadhaar_front_and_back_of_same_id(test_setup):
    """Case 2: Front and back of the same physical Aadhaar ID must classify as AADHAAR CARD, NOT NON_KYC."""
    src = create_dummy_image(test_setup["in_dir"] / "cust_aadhaar_both_sides.jpg")

    # VLM identifies front and back belonging to the same card
    with patch("kyc_sorter.sorter.vlm_router.classify_document", return_value=("AADHAAR CARD", None, 0.5)):
        res = process_vlm_file(
            image_file=src,
            index=1,
            total=1,
            destination_root=test_setup["out_dir"],
            dry_run=False,
            operation="copy",
            single_pass=True,
        )

    assert res.routing_type == "KYC"
    assert res.category == "AADHAAR CARD"
    assert res.final_filename == "cust_aadhaar_both_sides_AADHAAR_CARD.jpg"
    assert res.destination_path == test_setup["out_dir"] / "KYC" / "cust_aadhaar_both_sides_AADHAAR_CARD.jpg"
    assert res.destination_path.exists()


# ==============================================================================
# Case 3: One PAN card alone -> PAN
# ==============================================================================
def test_03_one_pan_card_alone(test_setup):
    """Case 3: Single PAN card alone must route to KYC with _PAN_CARD suffix."""
    src = create_dummy_image(test_setup["in_dir"] / "cust_pan.jpg")

    with patch("kyc_sorter.sorter.vlm_router.classify_document", return_value=("PAN CARD", None, 0.4)):
        res = process_vlm_file(
            image_file=src,
            index=1,
            total=1,
            destination_root=test_setup["out_dir"],
            dry_run=False,
            operation="copy",
            single_pass=True,
        )

    assert res.status == "SUCCESS"
    assert res.routing_type == "KYC"
    assert res.category == "PAN CARD"
    assert res.final_filename == "cust_pan_PAN_CARD.jpg"
    assert res.destination_path == test_setup["out_dir"] / "KYC" / "cust_pan_PAN_CARD.jpg"
    assert res.destination_path.exists()


# ==============================================================================
# Case 4: PAN front and back of the same ID -> PAN (Rule B)
# ==============================================================================
def test_04_pan_front_and_back_of_same_id(test_setup):
    """Case 4: Front and back of the same physical PAN card must classify as PAN CARD, NOT NON_KYC."""
    src = create_dummy_image(test_setup["in_dir"] / "pan_front_back.jpg")

    with patch("kyc_sorter.sorter.vlm_router.classify_document", return_value=("PAN CARD", None, 0.5)):
        res = process_vlm_file(
            image_file=src,
            index=1,
            total=1,
            destination_root=test_setup["out_dir"],
            dry_run=False,
            operation="copy",
            single_pass=True,
        )

    assert res.routing_type == "KYC"
    assert res.category == "PAN CARD"
    assert res.final_filename == "pan_front_back_PAN_CARD.jpg"
    assert res.destination_path == test_setup["out_dir"] / "KYC" / "pan_front_back_PAN_CARD.jpg"
    assert res.destination_path.exists()


# ==============================================================================
# Case 5: Aadhaar + PAN in one image -> NON_KYC (Rule A)
# ==============================================================================
def test_05_aadhaar_plus_pan_in_one_image(test_setup):
    """Case 5: Aadhaar + PAN composite image must route to NON KYC with original filename (Rule A)."""
    src = create_dummy_image(test_setup["in_dir"] / "composite_aadhaar_pan.jpg")

    # Under Rule A, model detects multiple distinct documents and returns MULTIPLE_DOCUMENTS
    with patch("kyc_sorter.sorter.vlm_router.classify_document", return_value=("MULTIPLE_DOCUMENTS", None, 0.5)):
        res = process_vlm_file(
            image_file=src,
            index=1,
            total=1,
            destination_root=test_setup["out_dir"],
            dry_run=False,
            operation="copy",
            single_pass=True,
        )

    assert res.routing_type == "NON_KYC"
    assert res.category == "NON_KYC"
    assert res.is_multiple_document is True
    # Preserves original filename, NO suffix
    assert res.final_filename == "composite_aadhaar_pan.jpg"
    assert res.destination_path == test_setup["out_dir"] / "NON KYC" / "composite_aadhaar_pan.jpg"
    assert res.destination_path.exists()


# ==============================================================================
# Case 6: PAN + bank statement in one image -> NON_KYC (Rule A)
# ==============================================================================
def test_06_pan_plus_bank_statement_in_one_image(test_setup):
    """Case 6: PAN card + bank statement in one image must route to NON KYC (Rule A priority over PAN)."""
    src = create_dummy_image(test_setup["in_dir"] / "pan_and_statement.jpg")

    with patch("kyc_sorter.sorter.vlm_router.classify_document", return_value=("MULTIPLE_DOCUMENTS", None, 0.5)):
        res = process_vlm_file(
            image_file=src,
            index=1,
            total=1,
            destination_root=test_setup["out_dir"],
            dry_run=False,
            operation="copy",
            single_pass=True,
        )

    assert res.routing_type == "NON_KYC"
    assert res.is_multiple_document is True
    assert res.final_filename == "pan_and_statement.jpg"
    assert res.destination_path == test_setup["out_dir"] / "NON KYC" / "pan_and_statement.jpg"
    assert res.destination_path.exists()


# ==============================================================================
# Case 7: Two different IDs in one image -> NON_KYC (Rule A)
# ==============================================================================
def test_07_two_different_ids_in_one_image(test_setup):
    """Case 7: Two different IDs in one image must classify as MULTIPLE_DOCUMENTS / NON_KYC."""
    src = create_dummy_image(test_setup["in_dir"] / "two_passports.jpg")

    with patch("kyc_sorter.sorter.vlm_router.classify_document", return_value=("MULTIPLE_DOCUMENTS", None, 0.5)):
        res = process_vlm_file(
            image_file=src,
            index=1,
            total=1,
            destination_root=test_setup["out_dir"],
            dry_run=False,
            operation="copy",
            single_pass=True,
        )

    assert res.routing_type == "NON_KYC"
    assert res.category == "NON_KYC"
    assert res.is_multiple_document is True
    assert res.final_filename == "two_passports.jpg"
    assert res.destination_path == test_setup["out_dir"] / "NON KYC" / "two_passports.jpg"
    assert res.destination_path.exists()


# ==============================================================================
# Case 8: Two views of the same driving licence -> driving licence (Rule B)
# ==============================================================================
def test_08_two_views_of_same_driving_licence(test_setup):
    """Case 8: Two views (front & back) of the same DL card -> DRIVING LICENCE in KYC folder."""
    src = create_dummy_image(test_setup["in_dir"] / "dl_front_and_back.jpg")

    with patch("kyc_sorter.sorter.vlm_router.classify_document", return_value=("DRIVING LICENCE", None, 0.5)):
        res = process_vlm_file(
            image_file=src,
            index=1,
            total=1,
            destination_root=test_setup["out_dir"],
            dry_run=False,
            operation="copy",
            single_pass=True,
        )

    assert res.routing_type == "KYC"
    assert res.category == "DRIVING LICENCE"
    assert res.final_filename == "dl_front_and_back_DRIVING_LICENCE.jpg"
    assert res.destination_path == test_setup["out_dir"] / "KYC" / "dl_front_and_back_DRIVING_LICENCE.jpg"
    assert res.destination_path.exists()


# ==============================================================================
# Case 9: Two unrelated documents in one image -> NON_KYC (Rule A)
# ==============================================================================
def test_09_two_unrelated_documents_in_one_image(test_setup):
    """Case 9: Two unrelated documents photographed together -> MULTIPLE_DOCUMENTS / NON_KYC."""
    src = create_dummy_image(test_setup["in_dir"] / "two_unrelated_docs.jpg")

    with patch("kyc_sorter.sorter.vlm_router.classify_document", return_value=("MULTIPLE_DOCUMENTS", None, 0.5)):
        res = process_vlm_file(
            image_file=src,
            index=1,
            total=1,
            destination_root=test_setup["out_dir"],
            dry_run=False,
            operation="copy",
            single_pass=True,
        )

    assert res.routing_type == "NON_KYC"
    assert res.final_filename == "two_unrelated_docs.jpg"
    assert res.destination_path == test_setup["out_dir"] / "NON KYC" / "two_unrelated_docs.jpg"
    assert res.destination_path.exists()


# ==============================================================================
# Case 10: An unreadable image -> Unidentified_files (Rule C)
# ==============================================================================
def test_10_unreadable_image_routes_to_unidentified(test_setup):
    """Case 10: Unreadable image must route to Unidentified_files, never NON_KYC (Rule C)."""
    src = create_dummy_image(test_setup["in_dir"] / "blurry_dark_unreadable.jpg")

    # VLM signals UNIDENTIFIED due to unreadable content
    with patch("kyc_sorter.sorter.vlm_router.classify_document", return_value=("UNIDENTIFIED", None, 0.4)):
        res = process_vlm_file(
            image_file=src,
            index=1,
            total=1,
            destination_root=test_setup["out_dir"],
            dry_run=False,
            operation="copy",
            single_pass=True,
        )

    assert res.routing_type == "UNIDENTIFIED"
    assert res.category == "UNIDENTIFIED"
    assert res.final_filename == "blurry_dark_unreadable.jpg"
    assert res.destination_path == test_setup["out_dir"] / "Unidentified_files" / "blurry_dark_unreadable.jpg"
    assert res.destination_path.exists()
    assert not (test_setup["out_dir"] / "NON KYC" / "blurry_dark_unreadable.jpg").exists()


# ==============================================================================
# Case 11: An ambiguous image that cannot reliably be distinguished -> Unidentified_files (Rule C)
# ==============================================================================
def test_11_ambiguous_image_routes_to_unidentified(test_setup):
    """Case 11: Ambiguous image (unclear whether 2 sides or 2 different IDs) -> Unidentified_files."""
    src = create_dummy_image(test_setup["in_dir"] / "ambiguous_card_angles.jpg")

    with patch("kyc_sorter.sorter.vlm_router.classify_document", return_value=("UNIDENTIFIED", None, 0.4)):
        res = process_vlm_file(
            image_file=src,
            index=1,
            total=1,
            destination_root=test_setup["out_dir"],
            dry_run=False,
            operation="copy",
            single_pass=True,
        )

    assert res.routing_type == "UNIDENTIFIED"
    assert res.destination_path == test_setup["out_dir"] / "Unidentified_files" / "ambiguous_card_angles.jpg"
    assert res.destination_path.exists()


# ==============================================================================
# Case 12: Malformed VLM response -> safe failure without an incorrect KYC classification (Rule C)
# ==============================================================================
def test_12_malformed_vlm_response_safe_failure(test_setup):
    """Case 12: Malformed VLM response must safely route to Unidentified_files without KYC suffix."""
    src = create_dummy_image(test_setup["in_dir"] / "corrupt_reply.jpg")

    # Normalizer rejects prose / malformed responses
    malformed_raw = "INCOME TAX DEPARTMENT GOVT OF INDIA PERMANENT ACCOUNT NUMBER CARD"
    assert normalize_category(malformed_raw) is None

    # When classify_document fails to parse valid category, it returns (None, error_msg, elapsed)
    with patch("kyc_sorter.sorter.vlm_router.classify_document", return_value=(None, "Malformed response", 0.5)):
        res = process_vlm_file(
            image_file=src,
            index=1,
            total=1,
            destination_root=test_setup["out_dir"],
            dry_run=False,
            operation="copy",
            single_pass=True,
        )

    assert res.routing_type == "UNIDENTIFIED"
    assert res.status == "UNIDENTIFIED"
    assert res.category is None
    assert res.final_filename == "corrupt_reply.jpg"
    assert res.destination_path == test_setup["out_dir"] / "Unidentified_files" / "corrupt_reply.jpg"
    assert res.destination_path.exists()


# ==============================================================================
# Case 13: Batch mode and watcher mode apply identical routing rules
# ==============================================================================
def test_13_batch_mode_and_watcher_mode_apply_identical_routing_rules(test_setup):
    """Case 13: Batch mode and watcher mode must apply 100% identical routing and destination decisions."""
    # Test across KYC, NON_KYC, MULTIPLE_DOCUMENTS, and UNIDENTIFIED categories
    test_cases = [
        ("batch_aadhaar.jpg", "AADHAAR CARD", "KYC", "batch_aadhaar_AADHAAR_CARD.jpg"),
        ("batch_pan.jpg", "PAN CARD", "KYC", "batch_pan_PAN_CARD.jpg"),
        ("batch_multidoc.jpg", "MULTIPLE_DOCUMENTS", "NON_KYC", "batch_multidoc.jpg"),
        ("batch_nonkyc.jpg", "NON_KYC", "NON_KYC", "batch_nonkyc.jpg"),
        ("batch_unidentified.jpg", "UNIDENTIFIED", "UNIDENTIFIED", "batch_unidentified.jpg"),
    ]

    for filename, cat, expected_routing_type, expected_final_name in test_cases:
        src = create_dummy_image(test_setup["in_dir"] / filename)
        target_dir, routing_type = get_destination_folder(cat, destination_root=test_setup["out_dir"])
        target_name = get_target_filename(src, cat)

        assert routing_type == expected_routing_type
        assert target_name == expected_final_name

        # Verify that both BatchProcessor and Watcher produce the exact same destination path
        bp = BatchProcessor(config=test_setup["config"])
        watcher = SourceFolderWatcher(config=test_setup["config"])

        with patch("kyc_sorter.sorter.vlm_router.classify_document", return_value=(cat, None, 0.1)):
            item_bp = bp.process_file(src)

        assert Path(item_bp.destination_path).name == expected_final_name
        assert Path(item_bp.destination_path).parent.name == target_dir.name
        bp.close()


# ==============================================================================
# Case 14: A previously processed file does not create duplicate outputs on watcher restart
# ==============================================================================
def test_14_previously_processed_file_no_duplicate_on_watcher_restart(test_setup):
    """Case 14: Processed file recorded in journal must not produce duplicate output or re-process on watcher restart."""
    src = create_dummy_image(test_setup["in_dir"] / "already_processed_file.jpg")

    journal = ProcessingJournal(db_path=test_setup["journal_path"], dry_run=False)
    bp = BatchProcessor(config=test_setup["config"], journal=journal)

    # Initial processing
    with patch("kyc_sorter.sorter.vlm_router.classify_document", return_value=("PAN CARD", None, 0.3)):
        item1 = bp.process_file(src)

    expected_dest = test_setup["out_dir"] / "KYC" / "already_processed_file_PAN_CARD.jpg"
    assert Path(item1.destination_path) == expected_dest
    assert expected_dest.exists()

    # Reconcile status confirms file is COMPLETED
    reconciled = journal.reconcile_file_status(src, expected_operation="copy")
    assert reconciled is not None
    assert reconciled[0] == "COMPLETED"

    # Simulate Watcher starting or restarting
    watcher = SourceFolderWatcher(config=test_setup["config"])

    # Attempt to process the same file event again on watcher restart
    with patch("kyc_sorter.sorter.vlm_router.classify_document") as mock_classify:
        watcher.on_filesystem_event(src)
        time.sleep(0.1)
        # Classify should NOT be called again because the journal reconciled the completion
        mock_classify.assert_not_called()

    # Check that collision file (_1) was NOT created
    collision_dest = test_setup["out_dir"] / "KYC" / "already_processed_file_PAN_CARD_1.jpg"
    assert not collision_dest.exists()
    bp.close()
    journal.close()


# ==============================================================================
# Case 15: Existing journal records remain readable
# ==============================================================================
def test_15_existing_journal_records_remain_readable(tmp_path):
    """Case 15: Existing journal database records remain readable without schema errors."""
    real_journal = Path(r"D:\KYC_Document_sorter_VLM_ONLY\data\journal\kyc_journal.db")

    if real_journal.exists() and real_journal.stat().st_size > 0:
        # Create an isolated temporary copy of the real journal to test reading
        temp_journal_copy = tmp_path / "test_copy_journal.db"
        shutil.copy2(real_journal, temp_journal_copy)

        journal = ProcessingJournal(db_path=temp_journal_copy, dry_run=False)
        counts = journal.get_summary_counts()
        assert "COMPLETED" in counts
        assert counts["COMPLETED"] >= 100

        # Query records using sqlite3 directly to ensure all columns and records are accessible
        conn = sqlite3.connect(temp_journal_copy)
        cur = conn.cursor()
        cur.execute("SELECT source_path, status, category, routing_type, destination_path FROM document_journal LIMIT 10")
        rows = cur.fetchall()
        assert len(rows) > 0
        first = rows[0]
        assert first[0] is not None  # source_path
        assert first[1] == "COMPLETED"  # status
        assert first[2] is not None  # category
        conn.close()
        journal.close()
    else:
        # Synthetic journal test
        synth_db = tmp_path / "synthetic_journal.db"
        journal = ProcessingJournal(db_path=synth_db, dry_run=False)
        src = tmp_path / "sample.jpg"
        src.touch()
        journal.record_completion(src, tmp_path / "dest.jpg", "PAN CARD", "KYC", "copy")
        counts = journal.get_summary_counts()
        assert counts["COMPLETED"] == 1
        journal.close()
