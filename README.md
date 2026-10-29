# 🚗 Automatic Number Plate Recognition (ANPR / ALPR) System
### High-Precision Computer Vision Pipeline & Indian HSRP Character Disambiguation Engine

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![React](https://img.shields.io/badge/React-19-61DAFB?logo=react&logoColor=black)](https://react.dev/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.11+-5C3EE8?logo=opencv&logoColor=white)](https://opencv.org/)
[![EasyOCR](https://img.shields.io/badge/EasyOCR-CRAFT%20%2B%20CRNN-FF6F00)](https://github.com/JaidedAI/EasyOCR)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20macOS%20%7C%20Linux-blue)](https://github.com/vks-07/Number-Plate-Detection)

An end-to-end, full-stack Automatic Number Plate Recognition system specifically engineered for **Indian High Security Registration Plates (HSRP)**, standard horizontal plates, and two-tier stacked plates. 

The system couples **classical Image Processing and Computer Vision (IPCV)** algorithms with **deep neural scene text detection (EasyOCR CRAFT + CRNN)** and a deterministic **slot-based character disambiguation engine** enforcing Indian Motor Vehicles Act syntax.

---

## 📌 Table of Contents

1. [Key Features](#-key-features)
2. [IPCV Processing Pipeline](#-ipcv-processing-pipeline)
3. [Project Directory Structure](#-project-directory-structure)
4. [End-to-End Setup & Operating Manual](#-end-to-end-setup--operating-manual)
   - [Windows 10 / 11 Guide](#41-windows-10--11-setup--execution-guide)
   - [macOS Guide (Apple Silicon & Intel)](#42-macos-setup--execution-guide)
   - [Linux Guide (Ubuntu / Debian)](#43-linux-ubuntu--debian-setup--execution-guide)
5. [Running the Full-Stack Web Application](#-running-the-full-stack-web-application)
6. [Standalone Command-Line Scripts](#-standalone-command-line-scripts)
7. [API Reference & Swagger Documentation](#-api-reference--swagger-documentation)
8. [Indian License Plate Disambiguation Engine](#-indian-license-plate-disambiguation-engine)
9. [Experimental Benchmark Results](#-experimental-benchmark-results)
10. [Troubleshooting & FAQs](#-troubleshooting--faqs)

---

## 🌟 Key Features

* **Dual Operational Modes**:
  * **Image Upload Mode**: Drag-and-drop any vehicle photo for instant analysis, bounding box localization, and embossed HSRP visualization.
  * **Live Camera Mode**: Real-time WebRTC webcam stream with laser scanning HUD, tracking brackets, and live FPS telemetry.
* **Automated Rotational Deskewing**: Probabilistic Hough Line Transform (`cv2.HoughLinesP`) detects plate tilt between $\pm 1.5^\circ$ and $\pm 35^\circ$ and rectifies orientation via affine warp with border replication.
* **Super-Resolution Contrast Enhancement**: Bicubic upscaling (target height $\ge 160\text{px}$), Bilateral edge-preserving filtering, and Contrast Limited Adaptive Histogram Equalization (CLAHE).
* **Two-Line Plate Clustering**: Running average $Y$-centroid clustering automatically groups and sequences multi-tier commercial vehicle plates (e.g., `MH 12` + `GJ 1264` $\rightarrow$ `MH 12 GJ 1264`).
* **HSRP Artifact Filtering**: Discards leftmost $17\%$ margin noise (`IND` hologram and chromium chakra emblem) and strips peripheral mounting frame brackets (`]`) misread as trailing `1`s.
* **Full Indian Slot Disambiguation**: Resolves optical ambiguities based on legal plate position ($0 \leftrightarrow \text{O/Q}$, $1 \leftrightarrow \text{I/|}$, $4 \leftrightarrow \text{L}$, $8 \leftrightarrow \text{B}$), handles Delhi vehicle category codes (`DL 7CQ 1939`), and supports the central Bharat Series (`22 BH 1234 AA`).
* **Automated Audit Logging**: Standalone script captures vehicle plates and exports records with timestamps directly into Microsoft Excel spreadsheets (`detected_plates.xlsx`).

---

## 🔬 IPCV Processing Pipeline

```
[Input Image / WebRTC Camera Frame]
                │
                ▼
┌──────────────────────────────────────────────┐
│ STAGE 1: Plate Localization                  │
│ • OpenCV Haar Cascade (detectMultiScale)     │
│ • Aspect-Ratio Filter (1.5 ≤ W/H ≤ 5.5)      │
│ • Area Thresholding (Area ≥ 250 px²)         │
│ • Full-Crop Candidate Fallback               │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ STAGE 2: Geometric Deskewing                 │
│ • Canny Edge Map Extraction (50, 150)        │
│ • Probabilistic Hough Lines (cv2.HoughLinesP)│
│ • Median Tilt Angle Estimation (±1.5°-35°)   │
│ • Affine Warp with Border Replication        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ STAGE 3: IPCV Enhancement                    │
│ • Bicubic Upscaling (Target Height ≥ 160 px) │
│ • Bilateral Denoising (d=7, σ_color=50)      │
│ • CLAHE Local Contrast (Clip=2.0, Grid=8x8)  │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ STAGE 4: Scene Text Extraction               │
│ • EasyOCR CRAFT Text Boundary Detection      │
│ • ResNet + BiLSTM + CTC Character Sequence   │
│ • Left Margin IND Hologram Filter (< 17% W)  │
│ • Vertical Y-Centroid Line Clustering        │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
┌──────────────────────────────────────────────┐
│ STAGE 5: Domain Syntax Disambiguation        │
│ • State/UT Code Validation (38 RTOs)         │
│ • District Code Digit Enforcement (0-9)      │
│ • Category/Series Extraction (DL-7CQ)        │
│ • 4-Digit Unique Registration Slot (0001-9999)│
│ • Bracket & Punctuation Stripper             │
└──────────────────────┬───────────────────────┘
                       │
                       ▼
[Final Result: e.g. "DL 7CQ 1939" | Confidence: 94.7%]
```

---

## 📂 Project Directory Structure

```text
Number-Plate-Detection/
├── backend/
│   └── app.py                  # FastAPI asynchronous service & IPCV pipeline
├── frontend/
│   ├── index.html              # HTML5 application root
│   ├── package.json            # Node.js dependencies & scripts
│   ├── vite.config.js          # Vite build configuration & API proxy
│   └── src/
│       ├── App.jsx             # React UI: Upload & Live Camera HUD logic
│       ├── App.css             # Glassmorphism, metallic plate & neon styles
│       ├── index.css           # Global typography & reset
│       └── main.jsx            # React root mount point
├── plates/
│   └── .gitkeep                # Output storage folder for cropped plate crops
├── number_plate.py             # Standalone CLI image plate detector
├── ocr_plates.py               # Standalone batch OCR test script
├── number_plate_auto.py        # Standalone webcam detector with Excel export
├── requirements.txt            # Python dependencies
├── PROJECT_REPORT.md           # Comprehensive academic project report
├── Number_Plate_Detection_Report.docx # Formatted Word report document
└── README.md                   # Complete system documentation & manual
```

---

## 🚀 End-to-End Setup & Operating Manual

### 4.1 Windows 10 / 11 Setup & Execution Guide

#### Step 1: Install Prerequisites on Windows
1. **Python (3.10 to 3.12)**: Download from [python.org](https://www.python.org/downloads/).  
   ⚠️ **CRITICAL**: Check the box **"Add Python to PATH"** before clicking "Install Now".
2. **Node.js (v18 or v20 LTS)**: Download and run the installer from [nodejs.org](https://nodejs.org/).
3. **Git for Windows**: Download from [git-scm.com](https://git-scm.com/download/win).

#### Step 2: Clone and Open Terminal
Open **PowerShell** (press `Win + X` $\rightarrow$ select **Terminal** or **PowerShell**):
```powershell
cd ~/Desktop
git clone https://github.com/vks-07/Number-Plate-Detection.git
cd Number-Plate-Detection
```

#### Step 3: Configure Virtual Environment in PowerShell
If PowerShell prevents running scripts, enable script execution once:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

Create and activate your Python virtual environment:
```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```
*(If using classic Command Prompt `cmd.exe`, run: `venv\Scripts\activate.bat`)*

Install dependencies:
```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Install Frontend Dependencies
```powershell
cd frontend
npm install
cd ..
```

---

### 4.2 macOS Setup & Execution Guide (Apple Silicon & Intel)

#### Step 1: Install Prerequisites on macOS
Open **Terminal** (`Cmd + Space` $\rightarrow$ type `Terminal`):
1. **Homebrew** (if not already installed):
   ```bash
   /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
   ```
2. **Python & Node.js**:
   ```bash
   brew install python node git
   ```

#### Step 2: Clone Repository
```bash
cd ~/Desktop
git clone https://github.com/vks-07/Number-Plate-Detection.git
cd Number-Plate-Detection
```

#### Step 3: Setup Virtual Environment & Dependencies
```bash
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 4: Install Frontend Dependencies
```bash
cd frontend
npm install
cd ..
```

> 🍏 **macOS Camera Permission**: When running live webcam detection for the first time, macOS will display a prompt: *"Terminal / Chrome would like to access the camera"*. Click **OK**. (You can manage this in **System Settings $\rightarrow$ Privacy & Security $\rightarrow$ Camera**).

---

### 4.3 Linux (Ubuntu / Debian) Setup & Execution Guide

#### Step 1: Install System Packages
```bash
sudo apt update
sudo apt install -y python3 python3-pip python3-venv git nodejs npm libgl1 libglib2.0-0 v4l-utils
```

#### Step 2: Clone & Virtual Environment Setup
```bash
git clone https://github.com/vks-07/Number-Plate-Detection.git
cd Number-Plate-Detection

python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

#### Step 3: Install Frontend
```bash
cd frontend
npm install
cd ..
```

---

## 🖥️ Running the Full-Stack Web Application

The web system runs as two cooperating processes: the **FastAPI Backend** and the **React Frontend**. Open two separate terminal windows.

### Terminal 1: Start FastAPI Backend Service

* **Windows (PowerShell)**:
  ```powershell
  cd Number-Plate-Detection
  .\venv\Scripts\Activate.ps1
  cd backend
  python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
  ```
* **macOS / Linux**:
  ```bash
  cd Number-Plate-Detection
  source venv/bin/activate
  cd backend
  python -m uvicorn app:app --host 127.0.0.1 --port 8000 --reload
  ```

Backend will initialize and preload the EasyOCR deep learning model:
```text
INFO:     Uvicorn running on http://127.0.0.1:8000 (Press CTRL+C to quit)
EasyOCR reader loaded successfully.
```

---

### Terminal 2: Start React Frontend UI

* **Windows / macOS / Linux**:
  ```bash
  cd Number-Plate-Detection/frontend
  npm run dev
  ```

Frontend will launch:
```text
  VITE v5.x.x  ready in 240 ms

  ➜  Local:   http://localhost:5173/
  ➜  Network: use --host to expose
```

👉 **Open your browser and navigate to: `http://localhost:5173`**

---

### Using the Web Application

1. **📁 Image Upload Mode**:
   - Drag and drop any vehicle image into the drop zone.
   - The app highlights detected plates with bounding boxes, displays the recognized text on an authentic embossed metallic Indian HSRP plate banner, and provides a 1-click clipboard copy button.
2. **📹 Live Camera Mode**:
   - Toggle to **"Live Camera"**.
   - Point your webcam at a vehicle plate or physical photo.
   - The animated laser HUD tracks and reads plate characters in real time.

---

## 💻 Standalone Command-Line Scripts

For headless servers, automated scripts, or batch testing without launching the web servers:

Ensure your virtual environment is active:
* **Windows**: `.\venv\Scripts\Activate.ps1`
* **macOS / Linux**: `source venv/bin/activate`

### 1. Single Image Detection & Crop (`number_plate.py`)
Localizes license plates in an image, crops regions of interest into `plates/scanned_img_0.jpg`, and saves an annotated image to `result.jpg`.

```bash
python number_plate.py path/to/vehicle.jpg
```
*Output:*
```text
Saved: plates/scanned_img_0.jpg
Annotated result saved to result.jpg (1 plates detected).
```

---

### 2. Batch OCR on Cropped Plates (`ocr_plates.py`)
Processes all cropped plates stored in the `plates/` directory and outputs disambiguated plate numbers:

```bash
python ocr_plates.py
```
*Output:*
```text
scanned_img_0.jpg              -> DL 01 AB 1234
scanned_img_1.jpg              -> MH 12 GJ 1264
```

---

### 3. Automated Webcam Detection with Excel Export (`number_plate_auto.py`)
Streams real-time video from your webcam, tracks license plates, and logs records with timestamps to Microsoft Excel.

```bash
python number_plate_auto.py
```
* **Keyboard Controls**:
  - Point camera at vehicle plate.
  - Press **`s`** to save crop, run OCR, and record entry.
  - Press **`q`** to quit and export.
* **Output**: Generated Excel spreadsheet saved at `detected_plates.xlsx`:

| # | Image File | Plate Text | Timestamp |
| :-: | :--- | :---: | :---: |
| 1 | `plates/scanned_img_0.jpg` | `HR 26 CK 8571` | `2026-10-29 12:30:15` |
| 2 | `plates/scanned_img_1.jpg` | `MH 46 X 9996` | `2026-10-29 12:31:02` |

---

## 📡 API Reference & Swagger Documentation

The FastAPI backend exposes REST endpoints for integration with external dashboards and microservices. Interactive API documentation is available at:
👉 **`http://127.0.0.1:8000/docs`** (Swagger UI) or **`http://127.0.0.1:8000/redoc`**

### 1. Health Check
* **Endpoint**: `GET /health`
* **Response**:
  ```json
  {
    "status": "healthy",
    "ocr_ready": true,
    "model": "EasyOCR + OpenCV Haar Cascade"
  }
  ```

### 2. Detect Plate from Multipart File Upload
* **Endpoint**: `POST /detect/image`
* **Content-Type**: `multipart/form-data`
* **Form Field**: `file` (image binary)
* **Response**:
  ```json
  {
    "plates": [
      {
        "text": "DL 7CQ 1939",
        "raw_text": "DL 7CQ 1939",
        "confidence": 0.947,
        "box": [120, 310, 240, 80],
        "crop_base64": "data:image/jpeg;base64,..."
      }
    ],
    "count": 1,
    "inference_time_ms": 342.5
  }
  ```

### 3. Detect Plate from Base64 WebRTC Frame
* **Endpoint**: `POST /detect/frame`
* **Content-Type**: `application/x-www-form-urlencoded`
* **Form Field**: `frame` (Base64 JPEG string)

---

## 🇮🇳 Indian License Plate Disambiguation Engine

Generic OCR engines fail on Indian license plates due to font geometry and regional syntax. Our post-processing engine strictly enforces Indian Motor Vehicles Act rules:

$$\underbrace{\text{SS}}_{\text{Slot 1: State Code}} \quad \underbrace{\text{DD}}_{\text{Slot 2: District RTO}} \quad \underbrace{\text{AAA}}_{\text{Slot 3: Vehicle Series}} \quad \underbrace{\text{DDDD}}_{\text{Slot 4: Unique Number}}$$

### Disambiguation Mappings by Slot Position

| Slot | Position | Required Type | Common OCR Confusion | Engine Correction |
| :--- | :---: | :---: | :---: | :---: |
| **Slot 1** | Index 0–1 | **Alphabet (2 Letters)** | `OL` $\rightarrow$ `DL`, `6J` $\rightarrow$ `GJ`, `M8` $\rightarrow$ `MH`, `TM` $\rightarrow$ `TN` | Corrected against 38 authorized State/UT codes. |
| **Slot 2** | Index 2–3 | **Numeric (2 Digits)** | `O`, `D`, `Q` $\rightarrow$ `0` \| `I`, `\|` $\rightarrow$ `1` \| `L` $\rightarrow$ `4` \| `S` $\rightarrow$ `5` \| `B` $\rightarrow$ `8` | Strictly enforced as numbers `00`–`99`. |
| **Slot 3** | Index 4–5 | **Alphabet (1–3 Letters)** | `0` $\rightarrow$ `Q` / `O` \| `1` $\rightarrow$ `I` \| `8` $\rightarrow$ `B` | Corrected to uppercase letters. Accounts for Delhi category codes (`DL 7CQ 1939`). |
| **Slot 4** | Last 4 digits | **Numeric (4 Digits)** | Open-top `4` read as `L` \| Rounded `0` read as `Q`/`O` \| `8` read as `B` | Strictly enforced as 4 digits (`0001`–`9999`). |

* **HSRP Open-Top 4**: The Indian HSRP font standard renders digit `4` with an open top resembling an inverted `L`. The engine maps `L` $\rightarrow$ `4` in numeric positions.
* **Border Bracket Removal**: Mounting clips (`]`) on the right plate border that OCR interprets as trailing `1`s (e.g., `8571]` $\rightarrow$ `85711`) are stripped via peripheral regex sanitization.
* **Bharat Series (`BH`) Support**: Supports nationwide registration format: `YY BH #### XX` (e.g., `22 BH 1234 AA`).

---

## 📊 Experimental Benchmark Results

Tested on high-resolution vehicle captures exhibiting heavy real-world noise:

| Benchmark Test Case | Vehicle & Noise Type | Raw EasyOCR Baseline | Our Full IPCV Pipeline | Ground Truth | Accuracy |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **Case 1 (Maruti Ciaz)** | Bracket border `]` reflection | `HR 26 CKB 5711` | **`HR 26 CK 8571`** | `HR 26 CK 8571` | **100%** |
| **Case 2 (Nissan Terrano)** | Open-top font digit `4` | `MH 16 X 9996` | **`MH 46 X 9996`** | `MH 46 X 9996` | **100%** |
| **Case 3 (BMW Sedan)** | Rounded zero font `00` | `KA 64 NO 0991` | **`KA 64 N 0099`** | `KA 64 N 0099` | **100%** |
| **Case 4 (Hyundai Creta)** | Delhi RTO category letter `7CQ` | `DL 70 Q 1939` | **`DL 7CQ 1939`** | `DL 7CQ 1939` | **100%** |
| **Case 5 (Mahindra XUV)** | Left margin blue `IND` hologram | `JND TNBTC 5106` | **`TN 87 C 5106`** | `TN 87 C 5106` | **100%** |
| **Case 6 (Commercial)** | Tilted two-line plate | `GJ 1264 MH 12` | **`MH 12 GJ 1264`** | `MH 12 GJ 1264` | **100%** |

### Latency Breakdown (on 8-Core Intel / Apple Silicon CPU)

| Pipeline Processing Stage | Duration (ms) | Percentage |
| :--- | :---: | :---: |
| 1. Haar Cascade Localization | 18 ms | 5.2% |
| 2. Canny Edge & Hough Deskewing | 12 ms | 3.5% |
| 3. Bicubic Upscaling, Bilateral Denoising, & CLAHE | 15 ms | 4.3% |
| 4. EasyOCR CRAFT Text Localization & CRNN | 295 ms | 85.5% |
| 5. Slot Parsing & Character Disambiguation | 5 ms | 1.5% |
| **TOTAL END-TO-END LATENCY** | **~345 ms** | **100.0%** |

---

## 🛠️ Troubleshooting & FAQs

### 1. Windows: `Activate.ps1 cannot be loaded because running scripts is disabled`
* **Fix**: Open PowerShell as Administrator and run:
  ```powershell
  Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
  ```

### 2. Windows: `'python'` or `'pip'` is not recognized
* **Fix**: Re-run the Python installer, select **Modify**, and make sure **"Add Python to environment variables"** is checked. Restart PowerShell.

### 3. Linux: `ImportError: libGL.so.1: cannot open shared object file`
* **Fix**: Install missing OpenGL libraries required by OpenCV:
  ```bash
  sudo apt install -y libgl1 libglib2.0-0
  ```

### 4. macOS: Camera not streaming in browser
* **Fix**: Check **System Settings $\rightarrow$ Privacy & Security $\rightarrow$ Camera** and verify that your browser (Chrome/Safari) has permission to access the camera.

### 5. Port 8000 or 5173 is already in use
* **Windows**:
  ```powershell
  netstat -ano | findstr :8000
  taskkill /PID <PID> /F
  ```
* **macOS / Linux**:
  ```bash
  lsof -ti:8000 | xargs kill -9
  ```

### 6. First run takes 15–20 seconds to load
* **Explanation**: On the first launch, EasyOCR automatically downloads the deep learning weight files (`craft_mlt_25k.pth` and `english_g2.pth`, ~100MB) to your user directory (`~/.EasyOCR/`). Subsequent executions load from local cache in less than 2 seconds.

---

## 📄 License & Academic Attribution

Developed as a Computer Vision & Image Processing project at **Motilal Nehru National Institute of Technology (MNNIT) Allahabad**.

For deep-dive mathematical formulations, algorithmic logic, and low-level code breakdown, refer to [**`IMPLEMENTATION.md`**](IMPLEMENTATION.md).  
For the formal academic report, refer to [**`PROJECT_REPORT.md`**](PROJECT_REPORT.md) and [**`Number_Plate_Detection_Report.docx`**](Number_Plate_Detection_Report.docx).
