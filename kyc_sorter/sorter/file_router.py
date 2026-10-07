"""
File Router and Segregation Engine.
Handles date-based folder partitioning, destination path calculation,
collision-safe filename deduplication, copy/move operations, and dry-run simulation.
"""

from __future__ import annotations

import datetime
import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

logger = logging.getLogger(__name__)


DEFAULT_FOLDERS_MAPPING: Dict[str, str] = {
    "AADHAAR_CARD": "AADHAAR CARD",
    "AADHAAR CARD": "AADHAAR CARD",
    "PAN_CARD": "PAN CARD",
    "PAN CARD": "PAN CARD",
    "VOTER_ID": "VOTER ID",
    "VOTER ID": "VOTER ID",
    "BANK_STATEMENT": "BANK STATEMENT",
    "BANK STATEMENT": "BANK STATEMENT",
    "BANK_PASSBOOK": "BANK STATEMENT",
    "CKYC_CONFIRMATION": "CKYC CONFIRMATION",
    "CKYC CONFIRMATION": "CKYC CONFIRMATION",
    "DRIVING_LICENCE": "DRIVING LICENCE",
    "DRIVING LICENCE": "DRIVING LICENCE",
    "DRIVING_LICENSE": "DRIVING LICENCE",
    "ELECTRICITY_BILL": "ELECTRICITY BILL",
    "ELECTRICITY BILL": "ELECTRICITY BILL",
    "GAS_BILL": "GAS BILL",
    "GAS BILL": "GAS BILL",
    "MUNICIPAL_TAX_RECEIPT": "MUNICIPAL TAX RECEIPT",
    "MUNICIPAL TAX RECEIPT": "MUNICIPAL TAX RECEIPT",
    "MUNICIPAL_RECEIPT": "MUNICIPAL TAX RECEIPT",
    "PHONE_BILL": "PHONE BILL",
    "PHONE BILL": "PHONE BILL",
    "RATION_CARD": "RATION CARD",
    "RATION CARD": "RATION CARD",
    "REGISTRATION_CERTIFICATE": "REGISTRATION CERTIFICATE",
    "REGISTRATION CERTIFICATE": "REGISTRATION CERTIFICATE",
    "PASSPORT": "PASSPORT",
    "GOVT_ID": "GOVT ID",
    "GOVT ID": "GOVT ID",
    "CKYC_DECLARATION": "CKYC DECLARATION",
    "CKYC DECLARATION": "CKYC DECLARATION",
    "IT_RETURNS": "IT RETURNS",
    "IT RETURNS": "IT RETURNS",
    "UNKNOWN": "NON KYC",
}


DEFAULT_FILENAME_SUFFIXES: Dict[str, str] = {
    "AADHAAR_CARD": "AADHAAR_CARD",
    "AADHAAR CARD": "AADHAAR_CARD",
    "AADHAAR": "AADHAAR_CARD",
    "PAN_CARD": "PAN_CARD",
    "PAN CARD": "PAN_CARD",
    "PAN": "PAN_CARD",
    "VOTER_ID": "VOTER_ID",
    "VOTER ID": "VOTER_ID",
    "BANK_STATEMENT": "BANK_STATEMENT",
    "BANK STATEMENT": "BANK_STATEMENT",
    "BANK_PASSBOOK": "BANK_STATEMENT",
    "CKYC_CONFIRMATION": "CKYC_CONFIRMATION",
    "CKYC CONFIRMATION": "CKYC_CONFIRMATION",
    "DRIVING_LICENCE": "DRIVING_LICENCE",
    "DRIVING LICENCE": "DRIVING_LICENCE",
    "DRIVING_LICENSE": "DRIVING_LICENCE",
    "ELECTRICITY_BILL": "ELECTRICITY_BILL",
    "ELECTRICITY BILL": "ELECTRICITY_BILL",
    "GAS_BILL": "GAS_BILL",
    "GAS BILL": "GAS_BILL",
    "MUNICIPAL_TAX_RECEIPT": "MUNICIPAL_RECEIPT",
    "MUNICIPAL TAX RECEIPT": "MUNICIPAL_RECEIPT",
    "MUNICIPAL_RECEIPT": "MUNICIPAL_RECEIPT",
    "PHONE_BILL": "PHONE_BILL",
    "PHONE BILL": "PHONE_BILL",
    "RATION_CARD": "RATION_CARD",
    "RATION CARD": "RATION_CARD",
    "REGISTRATION_CERTIFICATE": "REGISTRATION_CERTIFICATE",
    "REGISTRATION CERTIFICATE": "REGISTRATION_CERTIFICATE",
    "PASSPORT": "PASSPORT",
    "GOVT_ID": "GOVT_ID",
    "GOVT ID": "GOVT_ID",
    "CKYC_DECLARATION": "CKYC_DECLARATION",
    "CKYC DECLARATION": "CKYC_DECLARATION",
    "IT_RETURNS": "IT_RETURNS",
    "IT RETURNS": "IT_RETURNS",
    "UNKNOWN": "",
}


@dataclass
class RouteDecision:
    """Detailed result of a file routing operation."""
    source_path: Path
    destination_path: Path
    document_type: str
    folder_name: str
    date_folder: Optional[str]
    operation: str  # "copy" or "move"
    dry_run: bool
    success: bool
    filename_suffix: str = ""
    error: Optional[str] = None
    routing_time_sec: float = 0.0
    file_op_time_sec: float = 0.0
    destination_txt_path: Optional[Path] = None
    destination_json_path: Optional[Path] = None
    txt_routed: bool = False
    json_routed: bool = False
    txt_action: str = ""
    json_action: str = ""
    timing_logs: Dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Union[str, bool, None]]:
        return {
            "source_path": str(self.source_path),
            "destination_path": str(self.destination_path),
            "destination_txt_path": str(self.destination_txt_path) if self.destination_txt_path else None,
            "destination_json_path": str(self.destination_json_path) if self.destination_json_path else None,
            "document_type": self.document_type,
            "folder_name": self.folder_name,
            "date_folder": self.date_folder,
            "filename_suffix": self.filename_suffix,
            "operation": self.operation,
            "dry_run": self.dry_run,
            "success": self.success,
            "txt_routed": self.txt_routed,
            "json_routed": self.json_routed,
            "txt_action": self.txt_action,
            "json_action": self.json_action,
            "error": self.error,
        }


class FileRouter:
    """
    Manages routing of classified documents into segregated destination directories
    with automatic post-classification filename suffix renaming and collision prevention.
    Final structure (no subfolders, no date folders):
      - KYC documents: Destination files/KYC/<ORIGINAL_STEM>_<DOCUMENT_TYPE>.<ext>
      - Unknown documents: Destination files/NON KYC/<ORIGINAL_STEM>.<ext>
    """

    def __init__(
        self,
        output_directory: Union[str, Path],
        folders_mapping: Optional[Dict[str, str]] = None,
        suffixes_mapping: Optional[Dict[str, str]] = None,
        kyc_folder: str = "KYC",
        unknown_folder: str = "NON KYC",
        unknown_suffix: str = "",
        operation: str = "copy",
        create_date_folder: bool = False,
        create_subfolders: bool = False,
        date_format: str = "%Y-%m-%d",
        dry_run: bool = False,
        enable_renaming: bool = True,
    ):
        self.output_directory = Path(output_directory).resolve()
        self.folders_mapping = dict(DEFAULT_FOLDERS_MAPPING)
        if folders_mapping:
            for k, v in folders_mapping.items():
                self.folders_mapping[k.upper()] = str(v).strip()
        self.suffixes_mapping = dict(DEFAULT_FILENAME_SUFFIXES)
        if suffixes_mapping:
            for k, v in suffixes_mapping.items():
                self.suffixes_mapping[k.upper()] = str(v).strip()
        self.kyc_folder = kyc_folder
        self.unknown_folder = unknown_folder
        self.unknown_suffix = unknown_suffix
        self.operation = operation.strip().lower() if operation in ("copy", "move") else "copy"
        self.create_date_folder = create_date_folder
        self.create_subfolders = create_subfolders
        self.date_format = date_format
        self.dry_run = dry_run
        self.enable_renaming = enable_renaming

    def resolve_folder_name(self, document_type: str) -> str:
        """Map document type to configured destination folder name."""
        doc_key = (document_type or "").strip().upper()
        return self.folders_mapping.get(doc_key, self.unknown_folder)

    def resolve_suffix(self, document_type: str) -> str:
        """Map document type to configured post-classification filename suffix."""
        doc_key = (document_type or "").strip().upper()
        if doc_key in ("UNKNOWN", "NON_KYC", "FAILED", "IGNORED"):
            return ""
        return self.suffixes_mapping.get(doc_key, "")

    def calculate_synchronized_paths(
        self,
        source_file: Union[str, Path],
        document_type: str,
        processing_date: Optional[datetime.date] = None,
        rejection_category: Optional[str] = None,
    ) -> Tuple[Path, Path, Path, str, Optional[str]]:
        """
        Calculate collision-free target destination paths for the image and companion
        .txt and .json files, ensuring 100% synchronized basenames and collision counters.
        Rules:
          - KYC documents: assigned document-type suffix (<STEM>_<DOCUMENT_TYPE>.<ext>)
          - NON_KYC documents: preserve original filename exactly (<STEM>.<ext>)
        Returns: (candidate_img, candidate_txt, candidate_json, folder_name, date_folder_str)
        """
        src = Path(source_file)
        folder_name = self.resolve_folder_name(document_type)
        doc_key = (document_type or "").strip().upper()

        # Categorize directly under KYC or Unknown (No subfolders per requirements)
        is_non_kyc = (
            doc_key in ("UNKNOWN", "NON_KYC", "FAILED", "IGNORED")
            or folder_name.strip().upper() == self.unknown_folder.strip().upper()
            or (rejection_category is not None and rejection_category.upper() != "KYC")
        )

        if is_non_kyc:
            target_dir = self.output_directory / self.unknown_folder
            suffix_tag = ""
        else:
            target_dir = self.output_directory / self.kyc_folder
            suffix_tag = self.resolve_suffix(document_type) if self.enable_renaming else ""

        if self.create_subfolders:
            target_dir = target_dir / folder_name

        date_folder_str: Optional[str] = None
        if self.create_date_folder:
            d = processing_date or datetime.date.today()
            date_folder_str = d.strftime(self.date_format)
            target_dir = target_dir / date_folder_str

        # Post-classification renaming:
        # KYC: stem + "_" + suffix_tag
        # NON_KYC: preserve original filename (stem only)
        stem = src.stem
        ext = src.suffix
        if is_non_kyc or not suffix_tag:
            base_name = stem
        else:
            base_name = f"{stem}_{suffix_tag}"

        candidate_img = target_dir / f"{base_name}{ext}"
        candidate_txt = target_dir / f"{base_name}.txt"
        candidate_json = target_dir / f"{base_name}.json"

        # Synchronized collision prevention: if ANY of candidate image, .txt, or .json already exists,
        # increment synchronized counter so all three companion files share the exact same suffix
        if candidate_img.exists() or candidate_txt.exists() or candidate_json.exists():
            counter = 1
            while True:
                c_img = target_dir / f"{base_name}_{counter}{ext}"
                c_txt = target_dir / f"{base_name}_{counter}.txt"
                c_json = target_dir / f"{base_name}_{counter}.json"
                if not c_img.exists() and not c_txt.exists() and not c_json.exists():
                    candidate_img = c_img
                    candidate_txt = c_txt
                    candidate_json = c_json
                    break
                counter += 1

        return candidate_img, candidate_txt, candidate_json, folder_name, date_folder_str

    def calculate_destination_path(
        self,
        source_file: Union[str, Path],
        document_type: str,
        processing_date: Optional[datetime.date] = None,
        rejection_category: Optional[str] = None,
    ) -> Tuple[Path, str, Optional[str]]:
        """
        Calculate collision-free target destination path with post-classification renaming.
        Synchronized with companion .txt and .json outputs.
        Returns: (destination_path, folder_name, date_folder_str)
        """
        candidate_img, _, _, folder_name, date_folder_str = self.calculate_synchronized_paths(
            source_file=source_file,
            document_type=document_type,
            processing_date=processing_date,
            rejection_category=rejection_category,
        )
        return candidate_img, folder_name, date_folder_str

    def route(
        self,
        source_file: Union[str, Path],
        document_type: str,
        processing_date: Optional[datetime.date] = None,
        ocr_text: Optional[str] = None,
        json_data: Optional[Dict[str, Any]] = None,
        rejection_reason: Optional[str] = None,
        classification_status: Optional[str] = None,
        confidence: Optional[float] = None,
        rejection_category: Optional[str] = None,
        generate_companions: bool = True,
    ) -> RouteDecision:
        """
        Execute copy, move, or dry-run simulation for the given document and its companion
        .txt (OCR text) and .json (classification metadata) files.
        Guarantees:
          - Synchronized filenames and collision counters across image, .txt, and .json
          - Reuses source companion files when available
          - Generates missing companion files from in-memory OCR text and classification data
          - Zero silent file loss and zero overwrite
          - Full auditability for both KYC and NON_KYC files
        Returns RouteDecision with status.
        """
        src = Path(source_file)
        if not src.is_file():
            dest_dir = self.output_directory / self.unknown_folder
            return RouteDecision(
                source_path=src,
                destination_path=dest_dir / src.name,
                document_type=document_type,
                folder_name=self.unknown_folder,
                date_folder=None,
                operation=self.operation,
                dry_run=self.dry_run,
                success=False,
                error=f"Source file does not exist: {src}",
            )

        t_route0 = time.perf_counter()
        ts_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        t_name0 = time.perf_counter()
        dest_path, dest_txt, dest_json, folder_name, date_str = self.calculate_synchronized_paths(
            source_file=src,
            document_type=document_type,
            processing_date=processing_date,
            rejection_category=rejection_category,
        )
        naming_sec = time.perf_counter() - t_name0
        logger.info(
            "[FILENAME ASSIGNED] [%s] '%s' -> '%s' (naming time: %.4fs)",
            ts_now,
            src.name,
            dest_path.name,
            naming_sec,
        )

        doc_key = (document_type or "").strip().upper()
        is_non_kyc = (
            doc_key in ("UNKNOWN", "NON_KYC", "FAILED", "IGNORED")
            or folder_name.strip().upper() == self.unknown_folder.strip().upper()
            or (rejection_category is not None and rejection_category.upper() != "KYC")
        )
        suffix_tag = "" if is_non_kyc else (self.resolve_suffix(document_type) if self.enable_renaming else "")

        # Check for existing companion files in source directory
        src_txt = src.with_suffix(".txt")
        src_json = src.with_suffix(".json")
        has_src_txt = src_txt.is_file()
        has_src_json = src_json.is_file()

        # If source text file exists and in-memory ocr_text is None, read from source text
        if has_src_txt and ocr_text is None:
            try:
                ocr_text = src_txt.read_text(encoding="utf-8")
            except Exception as read_txt_err:
                logger.warning("Could not read existing source text '%s': %s", src_txt.name, read_txt_err)

        # If source JSON exists, load existing data to preserve any upstream backend schema
        source_json_content: Dict[str, Any] = {}
        if has_src_json:
            try:
                with open(src_json, "r", encoding="utf-8") as jf:
                    loaded = json.load(jf)
                    if isinstance(loaded, dict):
                        source_json_content = loaded
            except Exception as read_json_err:
                logger.warning("Could not parse existing source JSON '%s': %s", src_json.name, read_json_err)

        if self.dry_run:
            logger.info(
                "[DRY RUN] Would %s: '%s' -> '%s' | .txt -> '%s' | .json -> '%s'",
                self.operation,
                src.name,
                dest_path,
                dest_txt.name,
                dest_json.name,
            )
            return RouteDecision(
                source_path=src,
                destination_path=dest_path,
                destination_txt_path=dest_txt,
                destination_json_path=dest_json,
                document_type=document_type,
                folder_name=folder_name,
                date_folder=date_str,
                filename_suffix=suffix_tag,
                operation=self.operation,
                dry_run=True,
                success=True,
                txt_routed=True,
                json_routed=True,
                txt_action="WOULD_REUSE" if has_src_txt else "WOULD_GENERATE",
                json_action="WOULD_REUSE" if has_src_json else "WOULD_GENERATE",
                error=None,
                routing_time_sec=naming_sec,
                file_op_time_sec=0.0,
            )

        t_op0 = time.perf_counter()
        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)

            # 1. Main image file movement or copy
            if self.operation == "move":
                shutil.move(src, dest_path)
                logger.debug("Moved image: '%s' -> '%s'", src.name, dest_path)
            else:
                shutil.copy2(src, dest_path)
                logger.debug("Copied image: '%s' -> '%s'", src.name, dest_path)

            txt_action = "SKIPPED"
            txt_routed = False
            json_action = "SKIPPED"
            json_routed = False

            companion_errors: list[str] = []

            if generate_companions:
                t_comp0 = time.perf_counter()

                # 2. Companion .txt handling: Reuse or Generate
                try:
                    if has_src_txt:
                        if self.operation == "move":
                            shutil.move(src_txt, dest_txt)
                        else:
                            shutil.copy2(src_txt, dest_txt)
                        txt_action = "REUSED_SOURCE_TXT"
                    else:
                        # Write atomically via temp file to avoid partial/corrupted files
                        tmp_txt = dest_txt.with_name(f".{dest_txt.name}.tmp")
                        tmp_txt.write_text(ocr_text if ocr_text is not None else "", encoding="utf-8")
                        tmp_txt.replace(dest_txt)
                        txt_action = "GENERATED_FROM_OCR"
                    txt_routed = True
                except Exception as txt_err:
                    txt_action = f"FAILED: {txt_err}"
                    companion_errors.append(f"txt: {txt_err}")
                    logger.error("Failed to route/generate .txt companion '%s': %s", dest_txt.name, txt_err)

                # 3. Companion .json handling: Reuse or Generate
                try:
                    payload = {
                        "file": dest_path.name,
                        "source_filename": src.name,
                        "final_output_filename": dest_path.name,
                        "source_path": str(src.resolve()) if src.exists() else str(src),
                        "destination_path": str(dest_path.resolve()),
                        "document_type": document_type,
                        "confidence": round(confidence, 4) if confidence is not None else 0.0,
                        "classification_status": classification_status or ("IDENTIFIED" if document_type != "UNKNOWN" else "UNKNOWN"),
                        "rejection_category": rejection_category or ("KYC" if document_type != "UNKNOWN" else "NON_KYC"),
                        "rejection_reason": rejection_reason,
                        "processing_timestamp": ts_now,
                        "ocr_text_file": dest_txt.name,
                        "ocr_text": ocr_text if ocr_text is not None else "",
                        "image_path": str(dest_path.resolve()),
                        "json_path": str(dest_json.resolve()),
                    }
                    if source_json_content:
                        merged = dict(source_json_content)
                        merged.update(payload)
                        payload = merged
                    if json_data:
                        merged = dict(json_data)
                        merged.update(payload)
                        payload = merged

                    # Write atomically via temp file to avoid partial/corrupted JSON on crash
                    tmp_json = dest_json.with_name(f".{dest_json.name}.tmp")
                    with open(tmp_json, "w", encoding="utf-8") as jf:
                        json.dump(payload, jf, indent=2, ensure_ascii=False)
                    tmp_json.replace(dest_json)

                    if has_src_json and self.operation == "move" and src_json.exists():
                        try:
                            if src_json.resolve() != dest_json.resolve():
                                src_json.unlink()
                        except Exception:
                            pass
                        json_action = "REUSED_AND_UPDATED_SOURCE_JSON"
                    elif has_src_json:
                        json_action = "REUSED_AND_UPDATED_SOURCE_JSON"
                    else:
                        json_action = "GENERATED_FROM_CLASSIFIER"
                    json_routed = True
                except Exception as json_err:
                    json_action = f"FAILED: {json_err}"
                    companion_errors.append(f"json: {json_err}")
                    logger.error("Failed to route/generate .json companion '%s': %s", dest_json.name, json_err)

                comp_sec = time.perf_counter() - t_comp0
                logger.info(
                    "[COMPANION OUTPUTS] [%s] '%s' | .txt: %s ('%s') | .json: %s ('%s') (time: %.4fs)",
                    ts_now,
                    dest_path.name,
                    txt_action,
                    dest_txt.name,
                    json_action,
                    dest_json.name,
                    comp_sec,
                )

            file_op_sec = time.perf_counter() - t_op0
            total_routing_sec = time.perf_counter() - t_route0

            logger.info(
                "[FILE TRANSFER] [%s] %s complete for '%s' (file op time: %.4fs, total routing time: %.4fs)",
                ts_now,
                self.operation.upper(),
                dest_path.name,
                file_op_sec,
                total_routing_sec,
            )

            return RouteDecision(
                source_path=src,
                destination_path=dest_path,
                destination_txt_path=dest_txt if generate_companions else None,
                destination_json_path=dest_json if generate_companions else None,
                document_type=document_type,
                folder_name=folder_name,
                date_folder=date_str,
                filename_suffix=suffix_tag,
                operation=self.operation,
                dry_run=False,
                success=True,
                txt_routed=txt_routed,
                json_routed=json_routed,
                txt_action=txt_action,
                json_action=json_action,
                error="; ".join(companion_errors) if companion_errors else None,
                routing_time_sec=naming_sec,
                file_op_time_sec=file_op_sec,
            )

        except Exception as e:
            file_op_sec = time.perf_counter() - t_op0
            logger.error("Failed to route file '%s' -> '%s': %s", src.name, dest_path, e)
            return RouteDecision(
                source_path=src,
                destination_path=dest_path,
                destination_txt_path=dest_txt if generate_companions else None,
                destination_json_path=dest_json if generate_companions else None,
                document_type=document_type,
                folder_name=folder_name,
                date_folder=date_str,
                filename_suffix=suffix_tag,
                operation=self.operation,
                dry_run=False,
                success=False,
                error=str(e),
                routing_time_sec=naming_sec,
                file_op_time_sec=file_op_sec,
            )
