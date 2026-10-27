import os
import sys
import cv2
import numpy as np


def detect_and_crop_plate(
    img: np.ndarray,
    cascade_path: str,
    min_area: int = 500,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Args:
        img (np.ndarray): Input vehicle image matrix.
        cascade_path (str): File path to OpenCV Haar cascade XML.
        min_area (int): Minimum bounding box pixel area to accept.
    Returns:
        tuple[np.ndarray, list[np.ndarray]]: Annotated image and list of cropped plate regions.
    """
    plate_cascade = cv2.CascadeClassifier(cascade_path)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)

    annotated = img.copy()
    crops = []

    for (x, y, w, h) in plates:
        if w * h > min_area:
            cv2.rectangle(annotated, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(
                annotated, "Number Plate", (x, max(15, y - 5)),
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2
            )
            crops.append(img[y : y + h, x : x + w])

    return annotated, crops


def main():
    """
    Args:
        None
    Returns:
        None
    """
    image_path = sys.argv[1] if len(sys.argv) > 1 else "sample_plate.jpg"
    if not os.path.isfile(image_path):
        print(f"Error: Image '{image_path}' not found.")
        return

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: Unable to load '{image_path}'.")
        return

    cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    annotated, crops = detect_and_crop_plate(img, cascade_path)

    os.makedirs("plates", exist_ok=True)
    for idx, crop in enumerate(crops):
        save_path = f"plates/scanned_img_{idx}.jpg"
        cv2.imwrite(save_path, crop)
        print(f"Saved: {save_path}")

    cv2.imwrite("result.jpg", annotated)
    print(f"Annotated result saved to result.jpg ({len(crops)} plates detected).")


if __name__ == "__main__":
    main()
