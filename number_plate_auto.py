from datetime import datetime
import os
import sys
import cv2
import easyocr
from openpyxl import Workbook

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "backend"))
import app as backend_app
from app import extract_plate_text_advanced


def export_to_excel(results: list, output_path: str = "detected_plates.xlsx") -> None:
    """
    Args:
        results (list): Tuples of (filename, plate_text, timestamp).
        output_path (str): File path for generated Excel workbook.
    Returns:
        None
    """
    if not results:
        print("No plates saved. Skipping export.")
        return

    wb = Workbook()
    ws = wb.active
    ws.title = "Detected Plates"
    ws.append(["#", "Image File", "Plate Text", "Timestamp"])

    for idx, (filename, text, ts) in enumerate(results, start=1):
        ws.append([idx, filename, text, ts])

    for col in ws.columns:
        max_len = max(len(str(cell.value or "")) for cell in col)
        ws.column_dimensions[col[0].column_letter].width = max_len + 3

    wb.save(output_path)
    print(f"Exported {len(results)} records to {output_path}.")


def process_webcam_stream(cascade_path: str, min_area: int = 500) -> list:
    """
    Args:
        cascade_path (str): File path to OpenCV Haar cascade XML.
        min_area (int): Minimum pixel area for valid license plate.
    Returns:
        list: Captured plate records [(filename, plate_text, timestamp), ...].
    """
    cap = cv2.VideoCapture(0)
    cap.set(3, 640)
    cap.set(4, 480)

    reader = easyocr.Reader(["en"])
    plate_cascade = cv2.CascadeClassifier(cascade_path)
    os.makedirs("plates", exist_ok=True)

    records = []
    count = 0

    while True:
        success, img = cap.read()
        if not success:
            break

        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        plates = plate_cascade.detectMultiScale(img_gray, 1.1, 4)
        img_roi = None

        for (x, y, w, h) in plates:
            if w * h > min_area:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, "Number Plate", (x, y - 5), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (255, 0, 255), 2)
                img_roi = img[y : y + h, x : x + w]
                cv2.imshow("ROI", img_roi)

        cv2.imshow("Result", img)
        key = cv2.waitKey(1) & 0xFF

        if key == ord("s") and img_roi is not None:
            save_path = f"plates/scanned_img_{count}.jpg"
            cv2.imwrite(save_path, img_roi)

            backend_app.reader = reader
            backend_app.ocr_ready = True
            plate_text, _, _ = extract_plate_text_advanced(img_roi)
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            records.append((os.path.basename(save_path), plate_text, ts))
            count += 1
            print(f"[{ts}] Saved: {save_path} | Text: {plate_text}")

            cv2.rectangle(img, (0, 200), (640, 300), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, f"Saved: {plate_text}", (30, 265), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1.5, (0, 0, 255), 2)
            cv2.imshow("Result", img)
            cv2.waitKey(400)

        elif key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return records


def main():
    """
    Args:
        None
    Returns:
        None
    """
    cascade_path = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
    records = process_webcam_stream(cascade_path)
    export_to_excel(records)


if __name__ == "__main__":
    main()
