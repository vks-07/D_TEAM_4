import glob
import os
import sys
import cv2
import easyocr

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
import app as backend_app
from app import extract_plate_text_advanced


def extract_plate_text(image_path: str, reader: easyocr.Reader) -> str:
    """
    Args:
        image_path (str): File system path to the license plate image.
        reader (easyocr.Reader): Initialized EasyOCR reader instance.
    Returns:
        str: Extracted and disambiguated license plate characters.
    """
    backend_app.reader = reader
    backend_app.ocr_ready = True
    img = cv2.imread(image_path)
    if img is None:
        return ""
    text, _, _ = extract_plate_text_advanced(img)
    return text


def main():
    """
    Args:
        None
    Returns:
        None
    """
    reader = easyocr.Reader(["en"])
    plates_dir = "plates"
    image_paths = sorted(glob.glob(os.path.join(plates_dir, "*.jpg")))

    if not image_paths:
        print(f"No images found in '{plates_dir}/'.")
        return

    for path in image_paths:
        text = extract_plate_text(path, reader)
        print(f"{os.path.basename(path):30s} -> {text}")


if __name__ == "__main__":
    main()
