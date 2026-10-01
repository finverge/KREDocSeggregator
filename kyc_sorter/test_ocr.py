"""
Standalone PaddleOCR test utility.
Usage:
    python test_ocr.py [path/to/image.jpg]

If no image path is provided, creates a self-test synthetic sample
and verifies that PaddleOCR performs detection and recognition properly.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from PIL import Image, ImageDraw

# Ensure kyc_sorter and project root are in sys.path
_current_dir = Path(__file__).resolve().parent
_parent_dir = _current_dir.parent
for p in (str(_current_dir), str(_parent_dir)):
    if p not in sys.path:
        sys.path.insert(0, p)

try:
    from kyc_sorter.detector.ocr import OCREngine
except ImportError:
    from detector.ocr import OCREngine


def create_sample_image(output_path: Path) -> Path:
    """Create a high-contrast synthetic test card for standalone testing."""
    img = Image.new("RGB", (600, 200), color=(255, 255, 255))
    draw = ImageDraw.Draw(img)
    draw.text((20, 30), "INCOME TAX DEPARTMENT", fill=(0, 0, 0))
    draw.text((20, 70), "GOVT. OF INDIA", fill=(0, 0, 0))
    draw.text((20, 110), "Permanent Account Number", fill=(0, 0, 0))
    draw.text((20, 150), "ABCDE1234F", fill=(0, 0, 0))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)
    return output_path


def main():
    if len(sys.argv) > 1:
        image_path = Path(sys.argv[1])
    else:
        sample_path = _current_dir / "data" / "input" / "sample_test_card.png"
        print(f"No image path provided. Generating sample test card at '{sample_path}'...")
        image_path = create_sample_image(sample_path)

    print(f"Testing OCR on: {image_path.resolve()}")
    if not image_path.is_file():
        print(f"ERROR: Image file not found: {image_path}", file=sys.stderr)
        sys.exit(1)

    ocr_engine = OCREngine(language="en", use_gpu=False, cpu_threads=4)
    result = ocr_engine.extract_text(image_path)

    print("=" * 50)
    print("OCR TEST RESULTS")
    print("=" * 50)
    print(f"Status           : {'SUCCESS' if result.success else 'FAILED'}")
    if result.error:
        print(f"Error Details    : {result.error}")
    print(f"Lines Extracted  : {len(result.lines)}")
    print(f"Mean Confidence  : {result.mean_confidence:.2%}")
    print(f"Safe Text Preview: {result.get_safe_preview(max_chars=120)}")
    print("=" * 50)

    if not result.success or not result.text.strip():
        print("Test failed: No text was recognized.")
        sys.exit(1)

    print("Standalone OCR test passed successfully!")
    sys.exit(0)


if __name__ == "__main__":
    main()
