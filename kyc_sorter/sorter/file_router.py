"""
File Router and Segregation Engine.
Handles date-based folder partitioning, destination path calculation,
collision-safe filename deduplication, copy/move operations, and dry-run simulation.
"""

from __future__ import annotations

import datetime
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional, Union

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
    "UNKNOWN": "Unknown",
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
    "UNKNOWN": "UNKNOWN",
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

    def to_dict(self) -> Dict[str, Union[str, bool, None]]:
        return {
            "source_path": str(self.source_path),
            "destination_path": str(self.destination_path),
            "document_type": self.document_type,
            "folder_name": self.folder_name,
            "date_folder": self.date_folder,
            "filename_suffix": self.filename_suffix,
            "operation": self.operation,
            "dry_run": self.dry_run,
            "success": self.success,
            "error": self.error,
        }


class FileRouter:
    """
    Manages routing of classified documents into segregated destination directories
    with automatic post-classification filename suffix renaming and collision prevention.
    Final structure (no subfolders, no date folders):
      - KYC documents: Destination files/KYC/<ORIGINAL_STEM>_<DOCUMENT_TYPE>.<ext>
      - Unknown documents: Destination files/Unknown/<ORIGINAL_STEM>_UNKNOWN.<ext>
    """

    def __init__(
        self,
        output_directory: Union[str, Path],
        folders_mapping: Optional[Dict[str, str]] = None,
        suffixes_mapping: Optional[Dict[str, str]] = None,
        kyc_folder: str = "KYC",
        unknown_folder: str = "Unknown",
        unknown_suffix: str = "UNKNOWN",
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
        return self.suffixes_mapping.get(doc_key, self.unknown_suffix)

    def calculate_destination_path(
        self,
        source_file: Union[str, Path],
        document_type: str,
        processing_date: Optional[datetime.date] = None,
    ) -> Tuple[Path, str, Optional[str]]:
        """
        Calculate collision-free target destination path with post-classification renaming.
        Renaming pattern: ORIGINAL_FILENAME_WITHOUT_EXTENSION + "_" + DOCUMENT_TYPE_SUFFIX + ORIGINAL_EXTENSION
        Routing structure:
          - KYC: output_directory / KYC / <filename>
          - Unknown: output_directory / Unknown / <filename>
        Returns: (destination_path, folder_name, date_folder_str)
        """
        src = Path(source_file)
        folder_name = self.resolve_folder_name(document_type)
        suffix_tag = self.resolve_suffix(document_type) if self.enable_renaming else ""
        doc_key = (document_type or "").strip().upper()

        # Categorize directly under KYC or Unknown (No subfolders per requirements)
        if doc_key == "UNKNOWN" or folder_name.strip().upper() == self.unknown_folder.strip().upper():
            target_dir = self.output_directory / self.unknown_folder
        else:
            target_dir = self.output_directory / self.kyc_folder

        if self.create_subfolders:
            target_dir = target_dir / folder_name

        date_folder_str: Optional[str] = None
        if self.create_date_folder:
            d = processing_date or datetime.date.today()
            date_folder_str = d.strftime(self.date_format)
            target_dir = target_dir / date_folder_str

        # Post-classification renaming: stem + "_" + suffix_tag + extension
        stem = src.stem
        ext = src.suffix
        if suffix_tag:
            base_name = f"{stem}_{suffix_tag}"
        else:
            base_name = stem
        candidate_path = target_dir / f"{base_name}{ext}"

        # Collision prevention: if candidate file already exists, append _1, _2, etc.
        if candidate_path.exists():
            counter = 1
            while True:
                numbered_candidate = target_dir / f"{base_name}_{counter}{ext}"
                if not numbered_candidate.exists():
                    candidate_path = numbered_candidate
                    break
                counter += 1

        return candidate_path, folder_name, date_folder_str

    def route(
        self,
        source_file: Union[str, Path],
        document_type: str,
        processing_date: Optional[datetime.date] = None,
    ) -> RouteDecision:
        """
        Execute copy, move, or dry-run simulation for the given file.
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

        dest_path, folder_name, date_str = self.calculate_destination_path(
            source_file=src,
            document_type=document_type,
            processing_date=processing_date,
        )

        suffix_tag = self.resolve_suffix(document_type) if self.enable_renaming else ""

        if self.dry_run:
            logger.info("[DRY RUN] Would %s: '%s' -> '%s'", self.operation, src.name, dest_path)
            return RouteDecision(
                source_path=src,
                destination_path=dest_path,
                document_type=document_type,
                folder_name=folder_name,
                date_folder=date_str,
                filename_suffix=suffix_tag,
                operation=self.operation,
                dry_run=True,
                success=True,
                error=None,
            )

        try:
            dest_path.parent.mkdir(parents=True, exist_ok=True)
            if self.operation == "move":
                shutil.move(src, dest_path)
                logger.debug("Moved: '%s' -> '%s'", src.name, dest_path)
            else:
                shutil.copy2(src, dest_path)
                logger.debug("Copied: '%s' -> '%s'", src.name, dest_path)

            return RouteDecision(
                source_path=src,
                destination_path=dest_path,
                document_type=document_type,
                folder_name=folder_name,
                date_folder=date_str,
                filename_suffix=suffix_tag,
                operation=self.operation,
                dry_run=False,
                success=True,
                error=None,
            )
        except Exception as e:
            logger.error("Failed to route file '%s' -> '%s': %s", src.name, dest_path, e)
            return RouteDecision(
                source_path=src,
                destination_path=dest_path,
                document_type=document_type,
                folder_name=folder_name,
                date_folder=date_str,
                filename_suffix=suffix_tag,
                operation=self.operation,
                dry_run=False,
                success=False,
                error=str(e),
            )
