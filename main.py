"""
Root launcher for KYC Document Sorter.
Delegates execution to the kyc_sorter package.
"""

import sys
from pathlib import Path

# Add project root and package to sys.path
root_dir = Path(__file__).resolve().parent
pkg_dir = root_dir / "kyc_sorter"
for p in (str(root_dir), str(pkg_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

from kyc_sorter.main import main

if __name__ == "__main__":
    sys.exit(main())
