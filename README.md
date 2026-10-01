# KYC Document Sorter & Segregation Utility

An automated, privacy-first Python utility designed to continuously monitor, identify, verify, and segregate incoming bank KYC documents at scale using local PaddleOCR and multi-signal template classification.

---

## 1. Core Workflow & Operations

When incoming KYC image scans (Aadhaar, PAN, Voter ID, Driving Licence, Bank Statement, utility bills, etc.) arrive:
1. **Source Folder Monitoring**: Continuously monitors `Source Files` using a filesystem event listener (`watchdog`).
2. **File Stability & Readiness**: Verifies complete file writes, locks, and stable sizes before OCR processing begins.
3. **Event Deduplication**: Prevents duplicate processing across rapid OS filesystem events.
4. **Local PaddleOCR**: Extracts text without transmitting any data over the internet or cloud.
5. **Template Classification**: Classifies documents against KYC templates using multi-signal evidence without relying on original filenames.
6. **Destination Hierarchy**:
   - Supported KYC documents -> `Destination files/KYC/<DOCUMENT TYPE>/[YYYY-MM-DD/]`
   - Unrecognized / Non-KYC documents -> `Destination files/Unknown/[YYYY-MM-DD/]`
7. **Strict Filename Renaming**:
   - Format: `<ORIGINAL_NAME>_<DOCUMENT_TYPE>.<ext>`
   - Examples:
     - `IMG0001.jpg` -> `IMG0001_AADHAAR_CARD.jpg`
     - `IMG0002.jpg` -> `IMG0002_PAN_CARD.jpg`
     - `IMG0003.jpeg` -> `IMG0003_VOTER_ID.jpeg`
     - `IMG0004.png` -> `IMG0004_DRIVING_LICENCE.png`
     - `IMG0005.jpg` -> `IMG0005_UNKNOWN.jpg`
8. **Collision Safety**: Automatically appends `_1`, `_2` before the extension if a file already exists, guaranteeing no overwrites.
9. **Copy / Move Modes**: `copy` (default, source file preserved) and `move` (source file safely moved only after successful routing).
10. **Audit Logs & Security**: Generates CSV audit reports and terminal logs with customer PII strictly masked.

---

## 2. Quick Start for Non-Programmers

1. Double-click `run_sorter.bat` to launch the graphical desktop interface.
2. Select or verify the input (`Source Files`) and output (`Destination files`) folders.
3. Click **Run Batch Sorter** for one-time batch processing, or **Start Folder Watcher** for continuous real-time monitoring.

---

## 3. Quick Start for Developers (CLI)

### Batch Processing Mode
Process existing documents currently in `Source Files`:
```cmd
python main.py
```
Or specify custom paths / arguments:
```cmd
python main.py -i "Source Files" -o "Destination files"
```

### Continuous Event Watcher Mode
Monitor `Source Files` in real time for incoming documents:
```cmd
python main.py --watch
```
To also process existing files present in `Source Files` upon startup:
```cmd
python main.py --watch --process-existing
```

### Dry-Run Simulation Mode
Simulate detection and destination path calculation without modifying files:
```cmd
python main.py --dry-run
```

### Standalone OCR Test
Run PaddleOCR directly on any document image:
```cmd
python test_ocr.py path/to/image.jpg
```

### Run Automated Tests
Execute the complete test suite (38 unit and integration tests):
```cmd
python -m unittest discover -s kyc_sorter/tests -p "test_*.py"
```

---

## 4. Configuration (`config.yaml`)

Both source and destination folder paths, operation modes, date folder formatting, and document mappings are fully configurable in `config.yaml` without changing Python source code.
