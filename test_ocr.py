"""
Root launcher for standalone PaddleOCR test utility.
Usage:
    python test_ocr.py [path/to/image.jpg]
"""

import sys
from pathlib import Path

root_dir = Path(__file__).resolve().parent
pkg_dir = root_dir / "kyc_sorter"
for p in (str(root_dir), str(pkg_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from kyc_sorter.test_ocr import main

if __name__ == "__main__":
    main()
