"""
Automated PyInstaller build script for KYC Document Sorter.
Packages the application into a standalone Windows directory distribution.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from pathlib import Path

# Paths
PKG_DIR = Path(__file__).resolve().parent.parent
WORKSPACE_DIR = PKG_DIR.parent
DIST_DIR = WORKSPACE_DIR / "dist"
BUILD_DIR = WORKSPACE_DIR / "build"
SPEC_FILE = WORKSPACE_DIR / "KYC_Document_Sorter.spec"


def build():
    print("=" * 60)
    print("Building KYC Document Sorter Windows Distribution")
    print("=" * 60)

    # Check for PyInstaller
    try:
        import PyInstaller
        print(f"Found PyInstaller version: {PyInstaller.__version__}")
    except ImportError:
        print("PyInstaller not found. Installing PyInstaller via pip...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])

    cmd = [
        sys.executable,
        "-m",
        "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--windowed",
        "--name",
        "KYC_Document_Sorter",
        "--add-data",
        f"{PKG_DIR / 'config.yaml'};kyc_sorter",
        "--hidden-import",
        "rapidfuzz",
        "--hidden-import",
        "PIL",
        "--hidden-import",
        "yaml",
        "--hidden-import",
        "paddleocr",
        "--hidden-import",
        "paddle",
        "--hidden-import",
        "paddlex",
        "--hidden-import",
        "watchdog",
        "--hidden-import",
        "watchdog.observers",
        "--hidden-import",
        "watchdog.events",
        "--hidden-import",
        "tkinter",
        str(WORKSPACE_DIR / "main.py"),
    ]

    print("Running PyInstaller...")
    print("Command:", " ".join(cmd))
    res = subprocess.run(cmd, cwd=str(WORKSPACE_DIR))

    if res.returncode == 0:
        print("\nBuild completed successfully!")
        target_dist = DIST_DIR / "KYC_Document_Sorter"
        print(f"Standalone executable is located at: {target_dist / 'KYC_Document_Sorter.exe'}")

        # Copy auxiliary files into distribution
        for aux_src in (PKG_DIR / "config.yaml", WORKSPACE_DIR / "README.md", WORKSPACE_DIR / "run_sorter.bat"):
            if aux_src.is_file():
                shutil.copy2(aux_src, target_dist / aux_src.name)
                print(f"Copied {aux_src.name} -> {target_dist}")
        return 0
    else:
        print(f"\nBuild failed with exit code: {res.returncode}", file=sys.stderr)
        return res.returncode


if __name__ == "__main__":
    sys.exit(build())
