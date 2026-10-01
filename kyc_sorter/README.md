# KYC Document Sorter & Segregation Utility

An automated, privacy-first Python utility designed to identify, verify, and segregate incoming bank KYC documents at scale using local PaddleOCR and multi-signal template classification.

---

## 1. What the Utility Does

When banks transmit large batches of KYC image scans (Aadhaar, PAN, Voter ID, Passports, Driving Licences, utility bills, etc.), this application automatically:
1. Reads all incoming images from a configured **Source/Input folder**.
2. Runs **local PaddleOCR** to extract text without uploading any sensitive customer data to external cloud services or third-party APIs.
3. Classifies each document against **13 standardized KYC templates** using multi-signal evidence (strong keywords, regular keywords, mandatory gates, negative keyword vetoes, and regex patterns).
4. Routes images into **organized, date-stamped destination folders** (e.g., `Output/PAN/2026-09-30/IMG0002.jpg`).
5. Routes blurry, corrupt, non-KYC, and weak-evidence documents to a first-class **Unknown** directory.
6. Generates detailed **processing logs** and an **audit-ready CSV report** summarizing the status of every file.

---

## 2. How Document Detection Works

Document classification does **NOT** rely on filenames (e.g. a file named `IMG_Aadhaar.jpg` that actually contains a PAN card is correctly identified as a PAN card).

The classification engine analyzes multiple signals from the extracted OCR text:
- **Strong Keywords (+3 weight)**: Terms exclusive to a specific document type (e.g., *"Unique Identification Authority of India"* for Aadhaar, *"Permanent Account Number"* for PAN, *"Election Commission of India"* for Voter ID).
- **Regular Keywords (+1 weight)**: Supporting contextual words (e.g., *"Date of Birth"*, *"Father's Name"*, *"Address"*).
- **Regex Patterns (+2 weight)**: Document-specific format validators (e.g., 10-character PAN pattern `[A-Z]{5}[0-9]{4}[A-Z]`, 12-digit Aadhaar pattern, Voter EPIC number, MRZ passport strings, IFSC codes).
- **Negative Keywords (-2 weight)**: Exclusion rules that prevent cross-category misclassification (e.g., electricity bill terms immediately disqualify an image from being classified as an Aadhaar card).
- **Mandatory Keyword Gates**: Specialized high-precision categories (such as CKYC Confirmation, IT Returns, and Registration Certificates) score 0 unless category-defining mandatory terms are confirmed.
- **Space-Collapsed Fallback**: Handles low-resolution scans where OCR dropped inter-word spaces (e.g., `"uniqueidentificationauthorityofindia"`).

---

## 3. How PaddleOCR is Used

- **100% Local Inference**: Runs entirely on the local machine CPU. No KYC data, images, or extracted text leave your computer.
- **Engine Isolation**: The OCR layer is isolated in `detector/ocr.py`.
- **Fault-Tolerant Execution**: Corrupted, truncated, or unreadable files are safely intercepted per-file and routed to `Unknown` without stopping batch execution.

---

## 4. How Unknown Works

**UNKNOWN is a first-class category.** An image is routed to `Unknown` if:
- It is a non-KYC image (e.g. landscape photo, restaurant receipt, personal note).
- It is badly blurred, skewed, or unreadable.
- The OCR extraction produced no readable text.
- The evidence score is weak or the confidence falls below the configured threshold (default: 80%).

Weak evidence is never forced into a KYC category.

---

## 5. Supported Document Types

The utility supports all 13 document templates transferred from existing signatures:

| Document Type | Internal Key | Destination Folder | Filename Suffix | Example Output |
|---|---|---|---|---|
| Aadhaar Card | `AADHAAR_CARD` | `Aadhaar` | `Aadhar` | `IMG0001Aadhar.png` |
| PAN Card | `PAN_CARD` | `PAN` | `PAN` | `IMG0002PAN.jpg` |
| Voter ID Card | `VOTER_ID` | `Voter_ID` | `VoterID` | `IMG0003VoterID.jpeg` |
| Driving Licence | `DRIVING_LICENCE` | `Driving_Licence` | `DrivingLicence` | `IMG0004DrivingLicence.jpg` |
| Bank Passbook / Statement | `BANK_PASSBOOK` | `Bank_Passbook` | `BankPassbook` | `IMG0005BankPassbook.png` |
| Indian Passport | `PASSPORT` | `Passport` | `Passport` | `IMG0006Passport.jpg` |
| Electricity Bill | `ELECTRICITY_BILL` | `Electricity_Bill` | `ElectricityBill` | `IMG0007ElectricityBill.png` |
| Ration Card | `RATION_CARD` | `Ration_Card` | `RationCard` | `IMG0008RationCard.jpg` |
| Phone / Broadband Bill | `PHONE_BILL` | `Phone_Bill` | `PhoneBill` | `IMG0009PhoneBill.jpeg` |
| LPG Gas Bill | `GAS_BILL` | `Gas_Bill` | `GasBill` | `IMG0010GasBill.png` |
| CKYC Confirmation | `CKYC_CONFIRMATION` | `CKYC` | `CKYC` | `IMG0011CKYC.jpg` |
| Registration Certificate (RC) | `REGISTRATION_CERTIFICATE` | `Registration_Certificate` | `RegistrationCertificate` | `IMG0012RegistrationCertificate.png` |
| Income Tax Returns / Form 60 | `IT_RETURNS` | `IT_Returns` | `ITReturns` | `IMG0013ITReturns.pdf` |
| Unidentified / Non-KYC | `UNKNOWN` | `Unknown` | `Unknown` | `IMG0014Unknown.png` |

---

## 6. Project Structure

```
KYC_Document_Sorter/
|-- main.py                      # Primary entry point (CLI and GUI launcher)
|-- test_ocr.py                  # Standalone OCR validation utility
|-- config.yaml                  # Central configuration file
|-- run_sorter.bat               # Windows one-click desktop launcher
|-- README.md                    # Documentation
|
|-- detector/                    # Detection and OCR layer
|   |-- __init__.py
|   |-- ocr.py                   # PaddleOCR wrapper & error recovery
|   |-- templates.py             # 13 complete KYC template signatures
|   |-- classifier.py            # Multi-signal classification engine
|
|-- sorter/                      # Segregation & batch execution layer
|   |-- __init__.py
|   |-- file_router.py           # Path calculation, suffix renaming, date folders, collision handling
|   |-- batch_processor.py       # Batch orchestrator & CSV reporter
|   |-- watcher.py               # Watchdog event listener, safe write verifier & duplicate suppressor
|
|-- utils/                       # Supporting utilities
|   |-- __init__.py
|   |-- config_loader.py         # Config validation and path resolution
|   |-- logger.py                # Redacted, sanitized logging
|
|-- gui.py                       # Desktop graphical interface (Tkinter) with live watcher
|
|-- data/
|   |-- input/                   # Source directory for incoming images
|   |-- output/                  # Segregated date-partitioned output
|
|-- logs/                        # Runtime execution logs
|
|-- tests/                       # Unit and integration test suite
|   |-- test_config.py           # Configuration tests
|   |-- test_classifier.py       # Classifier tests across all templates
|   |-- test_router.py           # Routing, suffix renaming, date folders, and collision tests
|   |-- test_watcher.py          # Watchdog listener, file write check & duplicate prevention tests
|   |-- test_batch_pipeline.py   # End-to-end integration test
|
|-- deployment/                  # Windows packaging and distribution resources
    |-- README_DEPLOYMENT.md     # Deployment & air-gapped guide
    |-- build_exe.py             # Automated PyInstaller packaging script
```

---

## 7. Installation

1. Install Python 3.10+ (64-bit).
2. Install dependencies:
   ```cmd
   pip install -r requirements.txt
   ```

---

## 8. Configuration (`config.yaml`)

Configuration can be customized without altering any Python source code:

```yaml
application:
  name: "KYC Document Sorter"
  version: "1.0.0"

input:
  directory: "data/input"        # Where incoming images reside

output:
  directory: "data/output"      # Where sorted images will be placed
  operation: "copy"            # "copy" (safe default) or "move"
  create_date_folder: true     # Creates YYYY-MM-DD subfolders
  date_format: "%Y-%m-%d"

processing:
  confidence_threshold: 0.80   # Minimum confidence [0.0 - 1.0]
  dry_run: false              # When true: simulate without modifying files
  supported_extensions:
    - ".jpg"
    - ".jpeg"
    - ".png"
    - ".bmp"
    - ".tif"
    - ".tiff"

ocr:
  language: "en"
  use_gpu: false
  cpu_threads: 4

security:
  log_sensitive_text: false    # Redacts Aadhaar/PAN/Bank numbers from logs
  store_raw_ocr_debug: false   # Omit raw text dumps from CSV reports
```

---

## 9. Running the Utility

### A. Desktop GUI Mode (Recommended for non-programmers)
Simply double-click `run_sorter.bat` or run:
```cmd
python main.py --gui
```
- Select your Source and Destination folders using the **Browse** buttons.
- Select your mode (Copy, Move, or Dry Run).
- Click **Run Batch Sorter** for one-time folder scanning.
- Click **Start Folder Watcher** to activate real-time continuous monitoring on incoming documents.

### B. Command-Line (CLI) Modes

**1. One-Time Batch Processing:**
```cmd
python main.py -i "D:/Bank_KYC/Input" -o "D:/Bank_KYC/Output"
```

**2. Continuous Filesystem Event Watcher:**
```cmd
python main.py -i "D:/Bank_KYC/Input" -o "D:/Bank_KYC/Output" --watch
```
- Monitors the source directory continuously for newly added images using `watchdog`.
- Automatically verifies file-write completion and size stability before starting OCR.
- Suppresses duplicate processing across multiple OS events.
- Safely processes, renames, and routes incoming documents in real time.

---

## 10. Dry-Run Mode (Simulation)

To preview exactly how files would be classified and where they would be routed **before** copying or moving any files:
```cmd
python main.py --dry-run
```
Outputs:
```
[2026-09-30 15:30:20] [DRY RUN] Would copy: 'IMG0001.jpg' -> 'Output/Aadhaar/2026-09-30/IMG0001Aadhar.jpg'
```

---

## 11. Large Batch Processing & Fault Isolation

- **Memory Efficient**: Files are processed sequentially without loading the whole batch into RAM.
- **Fault-Isolated**: If file #243 is corrupt, unreadable, or invalid, it is logged and routed to `Unknown`. File #244 continues processing uninterrupted.

---

## 12. Output Structure & Post-Classification Renaming

Files are renamed using the pattern:
`ORIGINAL_FILENAME_WITHOUT_EXTENSION + DOCUMENT_TYPE_SUFFIX + ORIGINAL_EXTENSION`

And segregated into document-type folders partitioned by processing date:

```
Output/
|-- Aadhaar/
|   `-- 2026-09-30/
|       |-- IMG0001Aadhar.jpg
|       `-- IMG0002Aadhar.png
|-- PAN/
|   `-- 2026-09-30/
|       |-- IMG0003PAN.jpg
|       `-- IMG0003PAN_1.jpg    <-- duplicate collision safely resolved
|-- Voter_ID/
|   `-- 2026-09-30/
|       `-- IMG0004VoterID.jpeg
|-- Passport/
|   `-- 2026-09-30/
|       `-- IMG0005Passport.png
|-- Unknown/
|   `-- 2026-09-30/
|       |-- IMG0010Unknown.jpg
|       `-- blurred_scanUnknown.png
`-- processing_report_2026-09-30_153000.csv
```

**Collision Protection**: If an output file already exists, a collision number is safely appended (`IMG0003PAN_1.jpg`) so that no existing file is ever overwritten.

---

## 13. Logging & Privacy (PII Protection)

- Full logs are written to `logs/processing_YYYY-MM-DD_HHMMSS.log`.
- **Sensitive KYC Data Masking**: All 12-digit numbers (Aadhaar), 10-character PAN identifiers, and 9-18 digit account numbers are automatically masked in console outputs and log files (e.g. `XXXXX0000X`, `XXXX-XXXX-XXXX`).

---

## 14. Audit CSV Report

At the end of every batch, a timestamped CSV report is generated containing:
- `timestamp`: Time file was processed.
- `source_file`: Original filename.
- `source_path`: Full path to source file.
- `document_type`: Classified document type (or `UNKNOWN`).
- `confidence`: Confidence score (0.00 to 1.00).
- `status`: `IDENTIFIED`, `UNKNOWN`, or `FAILED`.
- `destination_path`: Where the segregated file was written.
- `error`: Details if an error or OCR failure occurred.

---

## 15. Standalone OCR Test Tool

To test OCR independently on any image:
```cmd
python test_ocr.py path/to/image.jpg
```
If run without an argument, it automatically generates a synthetic test card, executes OCR, and displays a sanitized text preview.

---

## 16. Running Automated Tests

Run the full automated unit and integration test suite:
```cmd
python -m unittest discover -s kyc_sorter/tests -p "test_*.py"
```
Tests verify configuration parsing, all 13 template classifications, router pathing, collision deduplication, date partitioning, dry-run mode, and end-to-end batch processing.
