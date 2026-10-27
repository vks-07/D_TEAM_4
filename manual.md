# 📖 User Manual & Operations Guide

A step-by-step guide to installing, configuring, starting, and operating the Automatic Number Plate Recognition (ANPR / ALPR) system.

---

## 📌 Table of Contents

1. [System Prerequisites](#1-system-prerequisites)
2. [Project Setup & Installation](#2-project-setup--installation)
3. [Starting the Application](#3-starting-the-application)
   - [Starting the Backend (FastAPI)](#step-1-start-the-fastapi-backend)
   - [Starting the Frontend (React + Vite)](#step-2-start-the-react-frontend)
4. [Using the Web Application](#4-using-the-web-application)
   - [Mode 1: Upload Image](#mode-1-upload-image)
   - [Mode 2: Live Camera](#mode-2-live-camera)
5. [Running Standalone CLI Scripts](#5-running-standalone-cli-scripts)
6. [API Reference & Testing](#6-api-reference--testing)
7. [Troubleshooting & Common Issues](#7-troubleshooting--common-issues)

---

## 1. System Prerequisites

Ensure the following tools are installed on your machine:

- **Python**: Version `3.9` to `3.12+` (`python3 --version`)
- **Node.js**: Version `18.0.0+` and `npm` (`node --version` and `npm --version`)
- **Webcam**: Built-in laptop camera or external USB webcam (for Live Camera mode)
- **Web Browser**: Google Chrome, Mozilla Firefox, Microsoft Edge, or Apple Safari

---

## 2. Project Setup & Installation

### Step 1: Open Terminal & Navigate to the Project

```bash
cd "/Users/varun/Desktop/ipcv project"
```

### Step 2: Set Up Python Virtual Environment

Create and activate a virtual environment to manage dependencies cleanly:

```bash
# Create virtual environment named 'venv'
python3 -m venv venv

# Activate on macOS / Linux:
source venv/bin/activate

# (On Windows, use: venv\Scripts\activate)
```

### Step 3: Install Backend Dependencies

Install the required Python packages into your virtual environment:

```bash
pip install -r requirements.txt
pip install "fastapi[standard]" python-multipart "opencv-python<5"
```

> [!IMPORTANT]
> Always install `opencv-python<5` (version 4.x). OpenCV 5.0 experimental builds alter the `cv2.CascadeClassifier` module structure.

### Step 4: Install Frontend Dependencies

Navigate to the `frontend/` directory and install the Node packages:

```bash
cd "/Users/varun/Desktop/ipcv project/frontend"
npm install
cd ..
```

---

## 3. Starting the Application

The full-stack application consists of two services running simultaneously:
1. **FastAPI Backend** on port **8000** (`http://localhost:8000`)
2. **React (Vite) Frontend** on port **5174** (`http://localhost:5174`)

---

### Step 1: Start the FastAPI Backend

Open a terminal tab and run:

```bash
cd "/Users/varun/Desktop/ipcv project/backend"
source ../venv/bin/activate
python -m uvicorn app:app --host 0.0.0.0 --port 8000
```

You should see output similar to:
```text
INFO:     Started server process
INFO:     Waiting for application startup.
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

- **Health Endpoint**: [http://localhost:8000/api/health](http://localhost:8000/api/health)
- **Interactive Swagger Docs**: [http://localhost:8000/docs](http://localhost:8000/docs)

---

### Step 2: Start the React Frontend

Open a **second terminal tab** and run:

```bash
cd "/Users/varun/Desktop/ipcv project/frontend"
npm run dev -- --host 0.0.0.0 --port 5174
```

You will see:
```text
  VITE v5.4.x  ready in 250 ms

  ➜  Local:   http://localhost:5174/
  ➜  Network: http://0.0.0.0:5174/
```

Now open **[http://localhost:5174](http://localhost:5174)** in your web browser.

---

## 4. Using the Web Application

The web interface features dual navigation tabs at the top: **Upload Image** and **Live Camera**.

```
+--------------------------------------------------------------+
| 🚗 Number Plate Detector                                     |
| [ 📁 Upload Image ]                 [ 📹 Live Camera ]       |
+--------------------------------------------------------------+
```

---

### Mode 1: Upload Image

1. Click on the **Upload Image** tab.
2. **Select an Image**:
   - Drag and drop any car image file onto the dashed dropzone area.
   - Or click anywhere inside the dropzone to browse your local filesystem (`JPG`, `PNG`, `WEBP`, `BMP`).
   - You can test with the provided sample images:
     - `sample_plate.jpg` (standard multi-line plate)
     - `tilted_2line_plate.jpg` (tilted plate)
3. Click the green **"Detect Plates & Read Text"** button.
4. **Inspect the Results**:
   - **Annotated Image**: Shows the full vehicle photo with green bounding boxes, cyan HUD corner brackets, and a floating text badge.
   - **Embossed License Plate Banner**: Displays the recognized alphanumeric characters styled with realistic Indian HSRP typography and the blue `IND` country strip.
   - **1-Click Copy**: Click the **Copy** button to copy the license plate string directly to your clipboard.
   - **Side-by-Side Crops**:
     - *Original Crop*: The raw bounding box cut from the vehicle.
     - *Enhanced IPCV Filter*: The deskewed, bilateral-filtered, CLAHE-enhanced image fed to the OCR engine.
   - **Bounding Box Coordinates**: Precise pixel coordinates `[x, y, width, height]` and normalized percentages.

---

### Mode 2: Live Camera

1. Click on the **Live Camera** tab.
2. Click the green **"Start Camera"** button.
3. When prompted by your browser, click **"Allow"** to grant camera permissions.
4. **Live Controls & HUD**:
   - **Live HUD Canvas**: Real-time bounding box and floating plate labels track moving plates directly over the video feed.
   - **Scanning Laser**: An animated green HUD laser sweeps continuously across the camera viewport.
   - **Live Metrics**: Live stream indicator, real-time FPS counter, and last hit timestamp badges.
   - **Live Tracking: ON / PAUSED**: Toggle background detection without stopping the video feed.
   - **Flip Camera**: Switch between front (user-facing) and back (environment) cameras on mobile or multi-camera devices.
   - **Stop Camera**: Shuts down the webcam stream and releases media device hardware locks.
5. **Live Detection History Feed**:
   - Cards appear in real-time below the camera whenever a license plate is recognized.
   - Each card displays the cropped plate, the embossed `IND` badge, match confidence, and timestamp.
   - Click **"Copy"** on any card to copy that vehicle's number.

---

## 5. Running Standalone CLI Scripts

For headless environments, server automation, or quick batch testing, four standalone CLI scripts are available in the project root:

### 1. Static Image Detection (`number_plate.py`)

Detects license plates in a photo, draws bounding boxes, crops plates into `plates/`, and outputs `result.jpg`.

```bash
source venv/bin/activate
python number_plate.py sample_plate.jpg
```
*Output:*
```text
Saved: plates/scanned_img_0.jpg
Annotated result saved to result.jpg (1 plates detected).
```

---

### 2. Batch OCR Extraction (`ocr_plates.py`)

Processes all cropped plates stored in the `plates/` directory with the full deskewing and disambiguation pipeline.

```bash
source venv/bin/activate
python ocr_plates.py
```
*Output:*
```text
scanned_img_0.jpg              -> DL 01 AB 1234
```

---

### 3. Real-Time Webcam Detection with Excel Export (`number_plate_auto.py`)

Opens your local webcam via OpenCV:
- Live green rectangles frame detected license plates in real-time.
- Press **`s`** on your keyboard to snapshot, run OCR, and record the plate.
- Press **`q`** to quit the stream and export all captures to an Excel spreadsheet (`detected_plates.xlsx`).

```bash
source venv/bin/activate
python number_plate_auto.py
```
*Output:*
```text
[2026-10-27 21:00:00] Saved: plates/scanned_img_0.jpg | Text: DL 01 AB 1234
Exported 1 records to detected_plates.xlsx.
```

---

### 4. Synthetic Vehicle Generator (`generate_test_image.py`)

Creates a synthetic vehicle graphic with a realistic 2-line license plate (`MH 07 BA 1234`) for automated testing.

```bash
source venv/bin/activate
python generate_test_image.py
```
*Output:*
```text
Created: test_car.jpg
```

---

## 6. API Reference & Testing

### 1. Health Check Endpoint

Check if the FastAPI service is healthy and the EasyOCR neural network model is loaded:

```bash
curl -s "http://localhost:8000/api/health"
```

**Response (JSON):**
```json
{
  "status": "ok",
  "ocr_ready": true,
  "ocr_loading": false,
  "cascade_model": "haarcascade_russian_plate_number.xml"
}
```

---

### 2. Detect & Recognize Endpoint

Upload a vehicle image file to extract bounding boxes, recognized text, and base64-encoded crops:

```bash
curl -X POST "http://localhost:8000/api/detect" \
  -F "image=@sample_plate.jpg" \
  -F "run_ocr=true"
```

**Response (JSON):**
```json
{
  "success": true,
  "plates": [
    {
      "bbox": [167, 248, 272, 91],
      "bbox_norm": [0.2783, 0.62, 0.4533, 0.2275],
      "text": "DL 01 AB 1234",
      "confidence": 0.972
    }
  ],
  "annotated_image": "data:image/jpeg;base64,...",
  "plate_images": ["data:image/jpeg;base64,..."],
  "processed_images": ["data:image/jpeg;base64,..."],
  "image_size": {
    "width": 600,
    "height": 400
  },
  "ocr_ready": true
}
```

---

## 7. Troubleshooting & Common Issues

### Issue 1: `Port 5000: Address already in use`
- **Cause**: On macOS, Apple's **AirPlay Receiver** service listens on port `5000` by default.
- **Solution**: The backend is configured to use port **`8000`** (`http://localhost:8000`). If port 8000 is occupied by another process, start Uvicorn on another port:
  ```bash
  python -m uvicorn app:app --host 0.0.0.0 --port 8001
  ```

---

### Issue 2: `OpenCV: not authorized to capture video` (Terminal Webcam Error)
- **Cause**: macOS enforces strict system privacy sandboxing on command-line terminal applications accessing the camera.
- **Solution**: Use the **Live Camera Mode** in the React web browser application at `http://localhost:5174`. The browser natively requests and manages camera permissions via the WebRTC API.

---

### Issue 3: `AttributeError: module 'cv2' has no attribute 'CascadeClassifier'`
- **Cause**: An experimental or incomplete OpenCV 5.x build was installed.
- **Solution**: Reinstall OpenCV 4.x in your virtual environment:
  ```bash
  pip install --force-reinstall "opencv-python<5"
  ```

---

### Issue 4: EasyOCR Pre-trained Model Weights
- **Information**: On the very first run, EasyOCR loads the CRAFT text detector (`craft_mlt_25k.pth`) and recognition model (`english_g2.pth`).
- Models are cached locally in your home directory at `~/.EasyOCR/model/` so subsequent startups load instantly from disk.

---

### Issue 5: Blurry or Low-Contrast Images
- **Tip**: Ensure the license plate is reasonably illuminated and not distorted by extreme motion blur.
- The built-in pipeline automatically applies **Bicubic Upscaling**, **Bilateral Smoothing**, and **CLAHE** to maximize readability on low-resolution or shadowed plates.

