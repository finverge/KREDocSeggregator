# Windows Deployment Guide — KYC Document Sorter

This guide outlines how to deploy the KYC Document Sorter on another Windows workstation (or standalone environment) without requiring manual Python installation or VS Code.

---

## 1. Overview of Runtime Architecture

The utility is designed to run completely standalone and offline:
- **Local OCR Engine**: PaddleOCR runs 100% locally on CPU (using OpenMP/MKL multi-threading). No documents or OCR texts are ever uploaded to cloud services.
- **Offline Model Cache**: Detection and recognition models are cached locally in `%USERPROFILE%\.paddlex\official_models` (or bundled with the distribution).
- **Self-Contained Configuration**: All paths, extensions, and confidence thresholds are configured via `config.yaml` or through the interactive graphical user interface (GUI).
- **Zero Hard-Coded Paths**: The application resolves paths relative to the current directory or the user's selected source/destination folders.

---

## 2. Option A: Standalone Windows Launcher (Recommended for Easy Deployment)

To distribute the application to employees with Python installed on Windows:

1. Copy the `kyc_sorter` directory to the target PC (e.g., `C:\KYC_Document_Sorter`).
2. Run `install_dependencies.bat` to install required packages (`paddleocr`, `paddlepaddle`, `rapidfuzz`, `pillow`, `pyyaml`).
3. Copy the pre-downloaded model cache `%USERPROFILE%\.paddlex\official_models` to the employee's user profile:
   ```cmd
   xcopy /E /I /Y official_models "%USERPROFILE%\.paddlex\official_models"
   ```
4. The user simply double-clicks `run_sorter.bat` to open the graphical desktop interface.

---

## 3. Option B: Packaging with PyInstaller as Windows Executable (.exe)

To bundle the application into a standalone Windows executable:

### Prerequisites:
Install PyInstaller:
```cmd
pip install pyinstaller
```

### Build Command:
Run the provided automated build script:
```cmd
python deployment\build_exe.py
```

This executes PyInstaller with:
```cmd
pyinstaller --noconfirm --onedir --windowed ^
    --name "KYC_Document_Sorter" ^
    --add-data "config.yaml;." ^
    --hidden-import "rapidfuzz" ^
    --hidden-import "PIL" ^
    --hidden-import "yaml" ^
    --hidden-import "paddleocr" ^
    --hidden-import "paddle" ^
    --hidden-import "paddlex" ^
    --hidden-import "tkinter" ^
    main.py
```

The resulting standalone distribution is written to `dist\KYC_Document_Sorter\`.

---

## 4. End-User Workflow (Non-Technical Employees)

Once deployed, employees do not need command-line or coding knowledge:

1. **Launch**: Double-click `run_sorter.bat` (or `KYC_Document_Sorter.exe`).
2. **Select Input**: Click **Browse** under "Input / Source Folder" and choose the folder where bank documents were received.
3. **Select Output**: Click **Browse** under "Output / Destination Folder" to choose where sorted files should be placed.
4. **Choose Mode**:
   - Keep **Copy** selected (default) to safely duplicate files.
   - Use **Dry Run** to inspect what would happen before modifying files.
5. **Start Processing**: Click **Start Processing**.
6. **Results**:
   - The live activity log and progress bar display status in real time.
   - Files are automatically routed into document-specific folders with the current date (e.g., `Output\Aadhaar\2026-09-30\`).
   - Unrecognized images and non-KYC files are routed to `Output\Unknown\2026-09-30\`.
   - A detailed CSV processing audit report (`processing_report_YYYY-MM-DD_HHMMSS.csv`) is generated automatically.

---

## 5. Troubleshooting Common Windows Issues

| Issue | Cause | Resolution |
|---|---|---|
| Missing DLL error (`VCRUNTIME140.dll`) | Visual C++ Redistributable missing | Install Microsoft Visual C++ 2015-2022 Redistributable (x64) from microsoft.com. |
| Model download on air-gapped PC | Target PC has no internet access | Pre-copy `.paddlex` directory into `%USERPROFILE%\.paddlex\`. |
| Long path error (`PathTooLongException`) | Deep nested directory paths | Enable Windows Long Path support in registry (`LongPathsEnabled = 1`). |
