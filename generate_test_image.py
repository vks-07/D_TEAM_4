import cv2
import numpy as np


def create_synthetic_test_image(output_path: str = "test_car.jpg") -> str:
    """
    Args:
        output_path (str): File destination path for generated image.
    Returns:
        str: Output file path.
    """
    img = np.zeros((480, 640, 3), dtype=np.uint8)

    for y in range(240):
        img[y, :] = [230 - y // 3, 200 - y // 4, 160 - y // 5]
    img[240:, :] = [80, 80, 80]

    cv2.rectangle(img, (180, 150), (460, 340), (140, 50, 20), -1)
    cv2.rectangle(img, (200, 100), (440, 170), (160, 70, 30), -1)
    cv2.rectangle(img, (210, 110), (310, 160), (200, 180, 160), -1)
    cv2.rectangle(img, (330, 110), (430, 160), (200, 180, 160), -1)

    cv2.circle(img, (230, 340), 30, (30, 30, 30), -1)
    cv2.circle(img, (410, 340), 30, (30, 30, 30), -1)

    plate_x, plate_y, plate_w, plate_h = 270, 280, 100, 40
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (255, 255, 255), -1)
    cv2.rectangle(img, (plate_x, plate_y), (plate_x + plate_w, plate_y + plate_h), (0, 0, 0), 2)

    cv2.putText(img, "MH 07", (plate_x + 8, plate_y + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
    cv2.putText(img, "BA 1234", (plate_x + 5, plate_y + 35), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

    cv2.imwrite(output_path, img)
    return output_path


if __name__ == "__main__":
    path = create_synthetic_test_image()
    print(f"Created: {path}")
