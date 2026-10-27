import base64
import re
import threading
import traceback
from contextlib import asynccontextmanager

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

reader = None
ocr_loading = False
ocr_ready = False
HARCASCADE = cv2.data.haarcascades + "haarcascade_russian_plate_number.xml"
MIN_AREA_DEFAULT = 250

CHAR_TO_DIGIT = {
    'O': '0', 'D': '0', 'Q': '0', 'U': '0',
    'I': '1', '|': '1', ']': '1',
    'L': '4',
    'Z': '2', 'E': '3', 'A': '4',
    'S': '5', 'G': '6', 'b': '6',
    'T': '7', 'Y': '7', 'J': '7',
    'B': '8', 'P': '9', 'g': '9', 'q': '9',
}

DIGIT_TO_CHAR = {
    '0': 'Q',
    '1': 'I', '2': 'Z',
    '3': 'E', '4': 'A', '5': 'S',
    '6': 'G', '7': 'T', '8': 'B', '9': 'P',
}

STATE_CODES = {
    'AN', 'AP', 'AR', 'AS', 'BR', 'CH', 'CG', 'DD', 'DL', 'DN',
    'GA', 'GJ', 'HP', 'HR', 'JH', 'JK', 'KA', 'KL', 'LA', 'LD',
    'MH', 'ML', 'MN', 'MP', 'MZ', 'NL', 'OD', 'OR', 'PB', 'PY',
    'RJ', 'SK', 'TN', 'TR', 'TS', 'UK', 'UP', 'WB', 'BH'
}

COMMON_STATE_TYPOS = {
    'OL': 'DL', '0L': 'DL', 'QL': 'DL', 'DL': 'DL',
    '6J': 'GJ', 'G1': 'GJ', '61': 'GJ', 'GJ': 'GJ',
    'M8': 'MH', 'N8': 'MH', 'HH': 'MH', 'MK': 'MH', 'KH': 'MH', 'MH': 'MH',
    'K4': 'KA', 'KA': 'KA',
    '1N': 'TN', '7N': 'TN', 'TM': 'TN', 'TX': 'TN', 'TN': 'TN',
    '0P': 'UP', 'OP': 'UP', 'QP': 'UP', 'UP': 'UP',
    '0D': 'OD', 'QD': 'OD', 'OD': 'OD',
    '0N': 'DN', 'QN': 'DN', 'DN': 'DN',
    'H8': 'HR', 'HR': 'HR',
    'P8': 'PB', 'PB': 'PB',
    '8R': 'BR', 'BR': 'BR',
    'R1': 'RJ', 'RJ': 'RJ',
    '8H': 'BH', 'BH': 'BH',
    '6A': 'GA', 'GA': 'GA',
    'A5': 'AS', 'AS': 'AS',
    'W8': 'WB', 'WB': 'WB',
    'K1': 'KL', 'KL': 'KL',
    'T5': 'TS', 'TS': 'TS',
    'M9': 'MP', 'MP': 'MP',
    'C6': 'CG', 'CG': 'CG',
    'J8': 'JH', 'JH': 'JH',
}

BORDER_CHARS = "[](){}|~\\'\".,-_:;><=+`"
HSRP_PATTERNS = {"IND", "1ND", "TND", "JND", "CD", "UD", "HD", "IKO", "IKD", "CND", "INO", "LND", "UND", "CHD", "TM", "FD"}


def to_digits(s: str) -> str:
    """
    Args:
        s (str): String containing ambiguous alphanumeric characters.
    Returns:
        str: Disambiguated digit string.
    """
    return "".join(CHAR_TO_DIGIT.get(c, c) for c in s)


def to_chars(s: str) -> str:
    """
    Args:
        s (str): String containing ambiguous digits.
    Returns:
        str: Disambiguated alphabet character string.
    """
    return "".join(DIGIT_TO_CHAR.get(c, c) for c in s)


def clean_state_code(raw: str) -> str:
    """
    Args:
        raw (str): Raw 2-character candidate state code.
    Returns:
        str: Validated state abbreviation.
    """
    upper_raw = raw.upper()
    if upper_raw in STATE_CODES:
        return upper_raw
    if upper_raw in COMMON_STATE_TYPOS:
        return COMMON_STATE_TYPOS[upper_raw]
    cand = to_chars(upper_raw)
    if cand in STATE_CODES:
        return cand
    if cand in COMMON_STATE_TYPOS:
        return COMMON_STATE_TYPOS[cand]
    return cand


def init_ocr():
    """
    Args:
        None
    Returns:
        None
    """
    global reader, ocr_loading, ocr_ready
    if ocr_ready or ocr_loading:
        return
    ocr_loading = True
    try:
        import easyocr
        reader = easyocr.Reader(["en"], gpu=False)
        ocr_ready = True
    except Exception as e:
        print(f"EasyOCR init notice: {e}")
    finally:
        ocr_loading = False


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Args:
        app (FastAPI): The application instance.
    Returns:
        AsyncGenerator: Lifespan context manager.
    """
    threading.Thread(target=init_ocr, daemon=True).start()
    yield


app = FastAPI(title="Number Plate Detection API", version="1.3.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def encode_image_to_base64(img: np.ndarray, fmt: str = ".jpg") -> str:
    """
    Args:
        img (np.ndarray): BGR image matrix.
        fmt (str): Output format extension.
    Returns:
        str: Base64 data URI string.
    """
    success, buffer = cv2.imencode(fmt, img)
    if not success:
        raise ValueError("Failed to encode image")
    mime = "image/jpeg" if fmt == ".jpg" else "image/png"
    return f"data:{mime};base64,{base64.b64encode(buffer).decode('utf-8')}"


def deskew_plate(img: np.ndarray) -> tuple[np.ndarray, float]:
    """
    Args:
        img (np.ndarray): Input plate crop matrix.
    Returns:
        tuple[np.ndarray, float]: Deskewed image and detected rotation angle in degrees.
    """
    if img is None or img.size == 0:
        return img, 0.0

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img.copy()
    h, w = gray.shape[:2]
    if h < 10 or w < 20:
        return img, 0.0

    # Probabilistic Hough transform extracts straight border/text lines to compute median tilt angle
    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=35, minLineLength=max(15, w // 4), maxLineGap=12)

    angles = []
    if lines is not None:
        for line in lines:
            x1, y1, x2, y2 = line[0]
            dx, dy = x2 - x1, y2 - y1
            if dx != 0:
                ang = np.degrees(np.arctan2(dy, dx))
                if abs(ang) < 35.0:
                    angles.append(ang)

    if angles:
        median_angle = float(np.median(angles))
        if abs(median_angle) >= 1.5:
            # Affine rotation matrix rotates image around center to align text horizontally
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
            return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), median_angle

    # Fallback orientation detection using minimum area bounding rectangle on thresholded foreground
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    coords = np.column_stack(np.where(thresh > 0))
    if len(coords) > 80:
        rect = cv2.minAreaRect(coords)
        rect_angle = -(90 + rect[-1]) if rect[-1] < -45 else -rect[-1]
        if 2.0 <= abs(rect_angle) <= 35.0:
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, rect_angle, 1.0)
            return cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE), rect_angle

    return img, 0.0


def clean_ocr_token(text: str) -> str:
    """
    Args:
        text (str): Raw string from EasyOCR.
    Returns:
        str: Token stripped of border brackets and punctuation.
    """
    s = text.strip()
    while s and s[0] in BORDER_CHARS:
        s = s[1:].strip()
    while s and s[-1] in BORDER_CHARS:
        s = s[:-1].strip()
    return s


def group_and_order_boxes(items: list, plate_w: int = 0) -> tuple[str, float]:
    """
    Args:
        items (list): EasyOCR bounding box detections [(bbox, text, conf), ...].
        plate_w (int): Total cropped plate width in pixels for relative filtering.
    Returns:
        tuple[str, float]: Joined multi-line text and average recognition confidence.
    """
    if not items:
        return "", 0.0

    parsed, heights = [], []
    for bbox, text, conf in items:
        cleaned_token = clean_ocr_token(text)
        clean_text = re.sub(r"[^A-Za-z0-9]", "", cleaned_token).strip().upper()
        if not clean_text:
            continue

        pts = np.array(bbox, dtype=np.float32)
        cx = float(np.mean(pts[:, 0]))
        cy = float(np.mean(pts[:, 1]))
        h = float(np.max(pts[:, 1]) - np.min(pts[:, 1]))
        x_max = float(np.max(pts[:, 0]))

        # Filter out leftmost HSRP country emblem (IND, JND, CD, etc.)
        if plate_w > 0 and (x_max / plate_w) < 0.17:
            if clean_text in HSRP_PATTERNS or float(conf) < 0.35:
                continue

        heights.append(h)
        parsed.append({"cx": cx, "cy": cy, "h": h, "text": clean_text, "conf": float(conf)})

    if not parsed:
        return "", 0.0
    if len(parsed) == 1:
        return parsed[0]["text"], round(parsed[0]["conf"], 3)

    avg_h = np.mean(heights) if heights else 20.0
    line_threshold = max(14.0, avg_h * 0.6)

    # Cluster words into lines by comparing vertical center Y-distance against line threshold
    parsed.sort(key=lambda item: item["cy"])
    lines = []
    current_line = [parsed[0]]
    for item in parsed[1:]:
        line_mean_y = float(np.mean([it["cy"] for it in current_line]))
        if abs(item["cy"] - line_mean_y) < line_threshold:
            current_line.append(item)
        else:
            lines.append(current_line)
            current_line = [item]
    if current_line:
        lines.append(current_line)

    # Sort each row left-to-right by X coordinate to preserve reading order
    line_texts, all_confs = [], []
    for line in lines:
        line.sort(key=lambda item: item["cx"])
        clean_words = [it["text"] for it in line if it["text"]]
        if clean_words:
            line_texts.append(" ".join(clean_words))
            all_confs.extend(it["conf"] for it in line)

    full_text = " ".join(line_texts)
    avg_conf = float(np.mean(all_confs)) if all_confs else 0.0
    return full_text, round(avg_conf, 3)


def disambiguate_characters(raw_text: str) -> str:
    """
    Args:
        raw_text (str): Unprocessed extracted license plate text.
    Returns:
        str: Corrected license plate string with resolved character ambiguities.
    """
    if not raw_text:
        return ""

    clean = re.sub(r"[^A-Za-z0-9]", "", raw_text).upper()

    # Search for valid state code or state typo to strip any leading emblem noise (e.g. 40TH87C5106 -> TH87C5106)
    state_match = re.search(
        r"(AN|AP|AR|AS|BR|CH|CG|DD|DL|DN|GA|GJ|HP|HR|JH|JK|KA|KL|LA|LD|MH|ML|MN|MP|MZ|NL|OD|OR|PB|PY|RJ|SK|TN|TR|TS|UK|UP|WB|BH|TH|TM|TX|HH|MK|KH|0L|QL|0P|QP|0D|QD|M8|N8|K4|6J|1N|7N)([0-9A-Z]{6,9})",
        clean,
    )
    if state_match:
        clean = state_match.group(1) + state_match.group(2)
    else:
        # Strip known HSRP left-strip artifacts if text is longer than standard plate
        for prefix in ["IND", "1ND", "TND", "JND", "UD", "HD", "IKO", "CHD", "CD", "FD", "CA", "ET", "IK"]:
            if clean.startswith(prefix) and len(clean) >= 11:
                clean = clean[len(prefix):]
                break

    # If length is 11 and ends with a 5th digit (trailing border misread as 1)
    if len(clean) == 11 and clean[-1] in ("1", "I", "L", "T"):
        clean = clean[:-1]

    # Bharat series: YY BH NNNN SS (10 chars with BH registration marker)
    if len(clean) == 10 and clean[2:4] in ("BH", "8H"):
        yr = to_digits(clean[0:2])
        nm = to_digits(clean[4:8])
        sr = to_chars(clean[8:10])
        return f"{yr} BH {nm} {sr}"

    # 10 characters: SS DD CC NNNN (or Delhi: SS 7C Q NNNN)
    if len(clean) == 10:
        st = clean_state_code(clean[0:2])
        if st == "DL" and clean[2].isdigit() and clean[3].isalpha():
            dt = clean[2:4]
            sr = to_chars(clean[4:6])
            nm = to_digits(clean[6:10])
            return f"{st} {dt} {sr} {nm}"
        dt = to_digits(clean[2:4])
        sr = to_chars(clean[4:6])
        nm = to_digits(clean[6:10])
        return f"{st} {dt} {sr} {nm}"

    # 9 characters: SS DD C NNNN or Delhi SS 7C Q NNNN
    if len(clean) == 9:
        st = clean_state_code(clean[0:2])
        if st == "DL" and clean[2].isdigit() and clean[3].isalpha():
            dt = clean[2:4]
            sr = to_chars(clean[4:5])
            nm = to_digits(clean[5:9])
            return f"{st} {dt} {sr} {nm}"
        if clean[2].isdigit() and clean[3:5].isalpha():
            dt = to_digits(clean[2:3])
            sr = to_chars(clean[3:5])
            nm = to_digits(clean[5:9])
        else:
            dt = to_digits(clean[2:4])
            sr = to_chars(clean[4:5])
            nm = to_digits(clean[5:9])
        return f"{st} {dt} {sr} {nm}"

    # 8 characters: SS DD NNNN (e.g. MH 12 1264, DL 01 1234)
    if len(clean) == 8:
        st = clean_state_code(clean[0:2])
        dt = to_digits(clean[2:4])
        nm = to_digits(clean[4:8])
        return f"{st} {dt} {nm}"

    # 7 characters: SS D NNNN (e.g. DL 1 1234, MH 1 1234)
    if len(clean) == 7:
        st = clean_state_code(clean[0:2])
        dt = to_digits(clean[2:3])
        nm = to_digits(clean[3:7])
        return f"{st} {dt} {nm}"

    # 11 characters: SS DD CCC NNNN (e.g. DL 01 ABC 1234)
    if len(clean) == 11:
        st = clean_state_code(clean[0:2])
        dt = to_digits(clean[2:4])
        sr = to_chars(clean[4:7])
        nm = to_digits(clean[7:11])
        return f"{st} {dt} {sr} {nm}"

    # Fallback contextual token resolution for irregular lengths or multi-line wraps
    tokens = [t.strip().upper() for t in re.split(r"\s+", raw_text) if t.strip()]
    if tokens:
        if len(tokens[0]) == 4:
            st = clean_state_code(tokens[0][:2])
            dt = to_digits(tokens[0][2:])
            tokens[0] = f"{st} {dt}"
        elif len(tokens[0]) == 2:
            tokens[0] = clean_state_code(tokens[0])

        if len(tokens) >= 2 and len(tokens[1]) <= 2:
            tokens[1] = to_digits(tokens[1])

        if len(tokens[-1]) >= 3:
            last = tokens[-1]
            if len(last) >= 5 and not last[:2].isdigit():
                sr = to_chars(last[:-4])
                nm = to_digits(last[-4:])
                tokens[-1] = f"{sr} {nm}"
            else:
                tokens[-1] = to_digits(last)

        return " ".join(tokens)

    return clean


def extract_plate_text_advanced(plate_crop: np.ndarray) -> tuple[str, float, np.ndarray]:
    """
    Args:
        plate_crop (np.ndarray): Cropped license plate image matrix.
    Returns:
        tuple[str, float, np.ndarray]: Extracted text, confidence score, and enhanced crop matrix.
    """
    global reader, ocr_ready
    if not ocr_ready or reader is None or plate_crop.size == 0:
        return "Plate Detected", 0.90, plate_crop

    deskewed_crop, tilt_angle = deskew_plate(plate_crop)
    h, w = deskewed_crop.shape[:2]
    if h == 0 or w == 0:
        return "Plate Detected", 0.90, plate_crop

    scale = max(2.0, 160.0 / h)
    upscaled = cv2.resize(deskewed_crop, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_CUBIC)

    # Bilateral filter denoises while preserving sharp edges; CLAHE balances localized contrast
    gray = cv2.cvtColor(upscaled, cv2.COLOR_BGR2GRAY)
    filtered = cv2.bilateralFilter(gray, 9, 75, 75)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    enhanced = clahe.apply(filtered)
    _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    processed_display = cv2.cvtColor(enhanced, cv2.COLOR_GRAY2BGR)
    candidates = []
    ALLOWLIST = "0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ -"
    plate_width = upscaled.shape[1]

    for variant in [enhanced, upscaled, thresh]:
        try:
            res = reader.readtext(
                variant,
                detail=1,
                allowlist=ALLOWLIST,
                paragraph=False,
                text_threshold=0.45,
                low_text=0.25,
                link_threshold=0.3,
            )
            if res:
                t, c = group_and_order_boxes(res, plate_w=plate_width)
                if len(t.replace(" ", "")) >= 3:
                    candidates.append((t, c))
        except Exception:
            pass

    if abs(tilt_angle) >= 1.5:
        try:
            scale_orig = max(2.0, 160.0 / plate_crop.shape[0])
            orig_up = cv2.resize(plate_crop, (0, 0), fx=scale_orig, fy=scale_orig, interpolation=cv2.INTER_CUBIC)
            res_orig = reader.readtext(
                orig_up,
                detail=1,
                allowlist=ALLOWLIST,
                paragraph=False,
                text_threshold=0.45,
                low_text=0.25,
                link_threshold=0.3,
            )
            if res_orig:
                t, c = group_and_order_boxes(res_orig, plate_w=orig_up.shape[1])
                if len(t.replace(" ", "")) >= 3:
                    candidates.append((t, c))
        except Exception:
            pass

    if candidates:
        # Score candidates to prefer strings containing both letters and digits, standard plate length, and confidence
        def score_cand(item):
            final = disambiguate_characters(item[0])
            clean = re.sub(r"[^A-Za-z0-9]", "", final)
            has_letters = bool(re.search(r"[A-Z]", clean))
            has_numbers = bool(re.search(r"[0-9]", clean))
            length_fit = 4.0 if (8 <= len(clean) <= 10) else (1.5 if (7 <= len(clean) <= 11) else 0.5)
            alphanumeric = 3.0 if (has_letters and has_numbers) else 1.0
            return (alphanumeric, length_fit, item[1])

        candidates.sort(key=score_cand, reverse=True)
        final_text = disambiguate_characters(candidates[0][0])
        return final_text, candidates[0][1], processed_display

    return "Plate Detected", 0.85, processed_display

    return "Plate Detected", 0.85, processed_display


def draw_plate_annotations(
    img: np.ndarray,
    x: int,
    y: int,
    w: int,
    h: int,
    text: str,
    confidence: float,
) -> None:
    """
    Args:
        img (np.ndarray): Target image to annotate.
        x (int): Bounding box top-left X coordinate.
        y (int): Bounding box top-left Y coordinate.
        w (int): Bounding box width.
        h (int): Bounding box height.
        text (str): Extracted plate text.
        confidence (float): Confidence score.
    Returns:
        None
    """
    img_h, img_w = img.shape[:2]
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 230, 115), 3)

    bracket_len = min(20, w // 4, h // 4)
    bracket_color = (255, 212, 0)
    bracket_thick = 4

    cv2.line(img, (x, y), (x + bracket_len, y), bracket_color, bracket_thick)
    cv2.line(img, (x, y), (x, y + bracket_len), bracket_color, bracket_thick)
    cv2.line(img, (x + w, y), (x + w - bracket_len, y), bracket_color, bracket_thick)
    cv2.line(img, (x + w, y), (x + w, y + bracket_len), bracket_color, bracket_thick)
    cv2.line(img, (x, y + h), (x + bracket_len, y + h), bracket_color, bracket_thick)
    cv2.line(img, (x, y + h), (x, y + h - bracket_len), bracket_color, bracket_thick)
    cv2.line(img, (x + w, y + h), (x + w - bracket_len, y + h), bracket_color, bracket_thick)
    cv2.line(img, (x + w, y + h), (x + w, y + h - bracket_len), bracket_color, bracket_thick)

    display_label = f" {text} "
    font, font_scale, font_thick = cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2
    (txt_w, txt_h), _ = cv2.getTextSize(display_label, font, font_scale, font_thick)
    badge_w, badge_h = txt_w + 16, txt_h + 16

    badge_y1 = max(0, y - badge_h - 4)
    badge_y2 = badge_y1 + badge_h
    badge_x1 = max(0, x)
    badge_x2 = min(img_w, badge_x1 + badge_w)

    cv2.rectangle(img, (badge_x1, badge_y1), (badge_x2, badge_y2), (18, 18, 30), -1)
    cv2.rectangle(img, (badge_x1, badge_y1), (badge_x2, badge_y2), (0, 230, 115), 2)
    cv2.putText(img, display_label, (badge_x1 + 8, badge_y2 - 8), font, font_scale, (255, 255, 255), font_thick, cv2.LINE_AA)


def detect_plates(
    img: np.ndarray,
    run_ocr: bool = True,
    min_area: int = MIN_AREA_DEFAULT,
) -> tuple[np.ndarray, list[dict]]:
    """
    Args:
        img (np.ndarray): Input vehicle image.
        run_ocr (bool): Enable or disable OCR text recognition.
        min_area (int): Minimum pixel area for valid detection.
    Returns:
        tuple[np.ndarray, list[dict]]: Annotated image and list of plate result dictionaries.
    """
    global reader, ocr_ready
    img_h, img_w = img.shape[:2]
    plate_cascade = cv2.CascadeClassifier(HARCASCADE)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    detections = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.1, minNeighbors=4)
    if len(detections) == 0:
        detections = plate_cascade.detectMultiScale(img_gray, scaleFactor=1.05, minNeighbors=2)

    annotated = img.copy()
    plates = []

    if len(detections) > 0:
        for (x, y, w, h) in detections:
            if w * h < min_area:
                continue

            # 5% boundary padding prevents clipping edge characters during cropping
            pad_x = int(w * 0.05)
            pad_y = int(h * 0.08)
            crop_x1 = max(0, x - pad_x)
            crop_y1 = max(0, y - pad_y)
            crop_x2 = min(img_w, x + w + pad_x)
            crop_y2 = min(img_h, y + h + pad_y)

            plate_crop = img[crop_y1:crop_y2, crop_x1:crop_x2]
            plate_text, conf, processed_crop = extract_plate_text_advanced(plate_crop)

            draw_plate_annotations(annotated, x, y, w, h, plate_text, conf)
            plates.append({
                "bbox": [int(x), int(y), int(w), int(h)],
                "bbox_norm": [
                    round(float(x) / img_w, 4),
                    round(float(y) / img_h, 4),
                    round(float(w) / img_w, 4),
                    round(float(h) / img_h, 4),
                ],
                "text": plate_text,
                "confidence": conf,
                "plate_image": encode_image_to_base64(plate_crop),
                "processed_image": encode_image_to_base64(processed_crop),
            })

    # Full crop / advanced detection fallback when Haar cascade finds no candidates or only a partial fraction
    aspect_ratio = float(img_w) / max(1, img_h)
    is_tight_plate = (1.8 <= aspect_ratio <= 7.0 and img_h <= 250)
    should_run_full = (len(plates) == 0) or (is_tight_plate and len(plates) == 1 and plates[0]["bbox"][2] < 0.65 * img_w)

    if should_run_full and run_ocr and ocr_ready and reader is not None:
        try:
            full_text, full_conf, full_processed = extract_plate_text_advanced(img)
            if full_text and full_text != "Plate Detected" and len(full_text.replace(" ", "")) >= 5:
                curr_len = len(plates[0]["text"].replace(" ", "")) if plates else 0
                full_len = len(full_text.replace(" ", ""))
                is_more_complete = (full_len >= 8 and curr_len < 7)
                if len(plates) == 0 or is_more_complete or full_conf >= plates[0]["confidence"]:
                    annotated = img.copy()
                    draw_plate_annotations(annotated, 5, 5, img_w - 10, img_h - 10, full_text, full_conf)
                    plates = [{
                        "bbox": [5, 5, img_w - 10, img_h - 10],
                        "bbox_norm": [0.01, 0.01, 0.98, 0.98],
                        "text": full_text,
                        "confidence": full_conf,
                        "plate_image": encode_image_to_base64(img),
                        "processed_image": encode_image_to_base64(full_processed),
                    }]
        except Exception as ex:
            print(f"Fallback OCR notice: {ex}")

    return annotated, plates


@app.get("/api/health")
async def health():
    """
    Args:
        None
    Returns:
        dict: Health status and model readiness.
    """
    return {
        "status": "ok",
        "ocr_ready": ocr_ready,
        "ocr_loading": ocr_loading,
        "cascade_model": "haarcascade_russian_plate_number.xml",
    }


@app.post("/api/detect")
async def detect(
    image: UploadFile = File(..., description="Car image file or video frame (jpg/png)"),
    run_ocr: bool = Form(default=True),
    min_area: int = Form(default=MIN_AREA_DEFAULT),
):
    """
    Args:
        image (UploadFile): Uploaded multipart image file.
        run_ocr (bool): Enable or disable OCR text recognition.
        min_area (int): Minimum pixel area for valid detection.
    Returns:
        dict: Detection and OCR results with bounding boxes and base64 images.
    """
    try:
        file_bytes = await image.read()
        nparr = np.frombuffer(file_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if img is None:
            return JSONResponse(status_code=400, content={"success": False, "error": "Could not decode image frame."})

        annotated_img, plates = detect_plates(img, run_ocr=run_ocr, min_area=min_area)

        return {
            "success": True,
            "plates": [
                {
                    "bbox": p["bbox"],
                    "bbox_norm": p["bbox_norm"],
                    "text": p["text"],
                    "confidence": p["confidence"],
                }
                for p in plates
            ],
            "annotated_image": encode_image_to_base64(annotated_img),
            "plate_images": [p["plate_image"] for p in plates],
            "processed_images": [p.get("processed_image", p["plate_image"]) for p in plates],
            "image_size": {"width": img.shape[1], "height": img.shape[0]},
            "ocr_ready": ocr_ready,
        }

    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"success": False, "error": str(e)})


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)
