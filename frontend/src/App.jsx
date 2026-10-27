import { useState, useRef, useCallback, useEffect } from 'react';
import './App.css';

function App() {
  const [mode, setMode] = useState('upload');

  const [selectedFile, setSelectedFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [results, setResults] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [copiedId, setCopiedId] = useState(null);
  const fileInputRef = useRef(null);

  const [isStreaming, setIsStreaming] = useState(false);
  const [cameraError, setCameraError] = useState(null);
  const [isDetectingLive, setIsDetectingLive] = useState(true);
  const [livePlates, setLivePlates] = useState([]);
  const [lastDetectedTime, setLastDetectedTime] = useState(null);
  const [facingMode, setFacingMode] = useState('environment');
  const [fps, setFps] = useState(0);

  const videoRef = useRef(null);
  const overlayCanvasRef = useRef(null);
  const captureCanvasRef = useRef(null);
  const streamRef = useRef(null);
  const isProcessingRef = useRef(false);
  const frameCountRef = useRef(0);
  const fpsTimerRef = useRef(null);

  const apiUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

  const copyToClipboard = (text, id) => {
    /*
    Args:
      text (string): License plate text to copy.
      id (string): Element identifier to trigger visual feedback.
    Returns:
      void
    */
    if (!text || text === 'Plate Detected' || text === 'NO TEXT') return;
    navigator.clipboard.writeText(text);
    setCopiedId(id);
    setTimeout(() => setCopiedId(null), 2000);
  };

  const stopCamera = useCallback(() => {
    /*
    Args:
      None
    Returns:
      void
    */
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((track) => track.stop());
      streamRef.current = null;
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null;
    }
    setIsStreaming(false);
    if (overlayCanvasRef.current) {
      const ctx = overlayCanvasRef.current.getContext('2d');
      ctx?.clearRect(0, 0, overlayCanvasRef.current.width, overlayCanvasRef.current.height);
    }
  }, []);

  const switchMode = (newMode) => {
    /*
    Args:
      newMode (string): Target mode ('upload' or 'camera').
    Returns:
      void
    */
    if (newMode === mode) return;
    if (mode === 'camera') {
      stopCamera();
    }
    setMode(newMode);
    setError(null);
    setCameraError(null);
  };

  useEffect(() => {
    return () => {
      stopCamera();
      if (fpsTimerRef.current) clearInterval(fpsTimerRef.current);
    };
  }, [stopCamera]);

  const handleFile = useCallback((file) => {
    /*
    Args:
      file (File): Selected image file object.
    Returns:
      void
    */
    if (file && file.type.startsWith('image/')) {
      setSelectedFile(file);
      setPreview(URL.createObjectURL(file));
      setResults(null);
      setError(null);
    } else {
      setError('Please select a valid image file (JPG, PNG, WebP).');
    }
  }, []);

  const handleDrag = useCallback((e) => {
    /*
    Args:
      e (DragEvent): Browser drag event.
    Returns:
      void
    */
    e.preventDefault();
    e.stopPropagation();
    if (e.type === 'dragenter' || e.type === 'dragover') {
      setDragActive(true);
    } else if (e.type === 'dragleave') {
      setDragActive(false);
    }
  }, []);

  const handleDrop = useCallback(
    (e) => {
      /*
      Args:
        e (DragEvent): Browser drop event.
      Returns:
        void
      */
      e.preventDefault();
      e.stopPropagation();
      setDragActive(false);
      if (e.dataTransfer.files && e.dataTransfer.files[0]) {
        handleFile(e.dataTransfer.files[0]);
      }
    },
    [handleFile]
  );

  const handleFileInput = useCallback(
    (e) => {
      /*
      Args:
        e (ChangeEvent): File input change event.
      Returns:
        void
      */
      if (e.target.files && e.target.files[0]) {
        handleFile(e.target.files[0]);
      }
    },
    [handleFile]
  );

  const handleSubmit = async () => {
    /*
    Args:
      None
    Returns:
      Promise<void>
    */
    if (!selectedFile) {
      setError('Please select an image first.');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('run_ocr', 'true');

      const response = await fetch(`${apiUrl}/api/detect`, {
        method: 'POST',
        body: formData,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status} ${response.statusText}`);
      }

      const data = await response.json();
      if (!data.success) {
        throw new Error(data.error || 'Detection failed.');
      }

      setResults(data);
    } catch (err) {
      setError(err.message || 'An error occurred while analyzing the image.');
    } finally {
      setLoading(false);
    }
  };

  const resetAll = () => {
    /*
    Args:
      None
    Returns:
      void
    */
    setSelectedFile(null);
    setPreview(null);
    setResults(null);
    setError(null);
    setLoading(false);
    if (fileInputRef.current) fileInputRef.current.value = '';
  };

  const startCamera = async () => {
    /*
    Args:
      None
    Returns:
      Promise<void>
    */
    setCameraError(null);
    try {
      if (streamRef.current) {
        stopCamera();
      }

      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: facingMode, width: { ideal: 640 }, height: { ideal: 480 } },
        audio: false,
      });
      streamRef.current = stream;

      if (videoRef.current) {
        videoRef.current.srcObject = stream;
        videoRef.current.onloadedmetadata = () => {
          videoRef.current.play();
          setIsStreaming(true);
        };
      }

      frameCountRef.current = 0;
      if (fpsTimerRef.current) clearInterval(fpsTimerRef.current);
      fpsTimerRef.current = setInterval(() => {
        setFps(frameCountRef.current);
        frameCountRef.current = 0;
      }, 1000);
    } catch (err) {
      console.error('Camera error:', err);
      setCameraError(
        err.name === 'NotAllowedError'
          ? 'Camera access denied. Please allow camera permissions.'
          : 'Unable to access camera.'
      );
      setIsStreaming(false);
    }
  };

  const toggleFacingMode = () => {
    /*
    Args:
      None
    Returns:
      void
    */
    const next = facingMode === 'environment' ? 'user' : 'environment';
    setFacingMode(next);
    if (isStreaming) {
      stopCamera();
      setTimeout(startCamera, 200);
    }
  };

  const drawOverlay = useCallback((plates, videoW, videoH) => {
    /*
    Args:
      plates (Array<Object>): Detected plate records.
      videoW (number): Video frame pixel width.
      videoH (number): Video frame pixel height.
    Returns:
      void
    */
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;

    if (canvas.width !== videoW || canvas.height !== videoH) {
      canvas.width = videoW;
      canvas.height = videoH;
    }

    const ctx = canvas.getContext('2d');
    if (!ctx) return;
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    plates.forEach((plate) => {
      let x, y, w, h;
      if (plate.bbox_norm) {
        x = plate.bbox_norm[0] * canvas.width;
        y = plate.bbox_norm[1] * canvas.height;
        w = plate.bbox_norm[2] * canvas.width;
        h = plate.bbox_norm[3] * canvas.height;
      } else if (plate.bbox) {
        [x, y, w, h] = plate.bbox;
      } else {
        return;
      }

      ctx.shadowColor = '#00ff88';
      ctx.shadowBlur = 10;
      ctx.strokeStyle = '#00ff88';
      ctx.lineWidth = 3;
      ctx.strokeRect(x, y, w, h);

      // Render HUD corner reticles at 4 vertices
      const cornerLen = Math.min(20, w / 4, h / 4);
      ctx.lineWidth = 4;
      ctx.strokeStyle = '#00d4ff';

      ctx.beginPath();
      ctx.moveTo(x, y + cornerLen);
      ctx.lineTo(x, y);
      ctx.lineTo(x + cornerLen, y);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(x + w - cornerLen, y);
      ctx.lineTo(x + w, y);
      ctx.lineTo(x + w, y + cornerLen);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(x, y + h - cornerLen);
      ctx.lineTo(x, y + h);
      ctx.lineTo(x + cornerLen, y + h);
      ctx.stroke();

      ctx.beginPath();
      ctx.moveTo(x + w - cornerLen, y + h);
      ctx.lineTo(x + w, y + h);
      ctx.lineTo(x + w, y + h - cornerLen);
      ctx.stroke();

      ctx.shadowBlur = 0;

      // Position floating badge above box or clamp inside viewport boundary
      const displayText = plate.text && plate.text !== 'Plate Detected' ? plate.text : 'NUMBER PLATE';
      ctx.font = 'bold 15px monospace';
      const textMetrics = ctx.measureText(displayText);
      const badgeW = textMetrics.width + 26;
      const badgeH = 26;
      const badgeY = Math.max(0, y - badgeH - 6);

      ctx.fillStyle = '#0f172a';
      ctx.fillRect(x, badgeY, badgeW, badgeH);
      ctx.strokeStyle = '#00ff88';
      ctx.lineWidth = 1.5;
      ctx.strokeRect(x, badgeY, badgeW, badgeH);

      ctx.fillStyle = '#0066ff';
      ctx.fillRect(x + 2, badgeY + 2, 6, badgeH - 4);

      ctx.fillStyle = '#00ff88';
      ctx.fillText(displayText, x + 14, badgeY + 18);
    });
  }, []);

  // Frame processing loop with lock to prevent overlapping network requests
  useEffect(() => {
    let timerId = null;

    const processFrame = async () => {
      if (!isStreaming || !isDetectingLive || isProcessingRef.current) {
        timerId = setTimeout(processFrame, 200);
        return;
      }

      const video = videoRef.current;
      const captureCanvas = captureCanvasRef.current;
      if (!video || video.readyState < 2 || video.videoWidth === 0 || !captureCanvas) {
        timerId = setTimeout(processFrame, 200);
        return;
      }

      isProcessingRef.current = true;
      const vW = video.videoWidth;
      const vH = video.videoHeight;
      captureCanvas.width = vW;
      captureCanvas.height = vH;
      captureCanvas.getContext('2d').drawImage(video, 0, 0, vW, vH);

      captureCanvas.toBlob(
        async (blob) => {
          if (!blob) {
            isProcessingRef.current = false;
            timerId = setTimeout(processFrame, 200);
            return;
          }

          try {
            const formData = new FormData();
            formData.append('image', blob, 'frame.jpg');
            formData.append('run_ocr', 'true');

            const res = await fetch(`${apiUrl}/api/detect`, { method: 'POST', body: formData });
            if (res.ok) {
              const data = await res.json();
              if (data.success) {
                frameCountRef.current += 1;
                drawOverlay(data.plates || [], vW, vH);

                if (data.plates && data.plates.length > 0) {
                  setLastDetectedTime(new Date().toLocaleTimeString());
                  setLivePlates((prev) => {
                    const newEntries = data.plates.map((p, idx) => ({
                      id: `${Date.now()}-${idx}`,
                      text: p.text,
                      confidence: p.confidence,
                      image: data.plate_images ? data.plate_images[idx] : null,
                      time: new Date().toLocaleTimeString(),
                    }));
                    return [...newEntries, ...prev].slice(0, 6);
                  });
                }
              }
            }
          } catch (err) {
            // Ignore transient frame request errors
          } finally {
            isProcessingRef.current = false;
            timerId = setTimeout(processFrame, 250);
          }
        },
        'image/jpeg',
        0.85
      );
    };

    if (isStreaming && isDetectingLive) {
      timerId = setTimeout(processFrame, 300);
    }

    return () => {
      if (timerId) clearTimeout(timerId);
    };
  }, [isStreaming, isDetectingLive, drawOverlay, apiUrl]);

  return (
    <div className="app">
      <header className="header">
        <div className="logo">
          <span className="logo-icon">🚗</span>
          <h1>Number Plate Detector</h1>
        </div>
        <p className="subtitle">Automatic License Plate Detection & High-Accuracy OCR Text Recognition</p>
      </header>

      <nav className="mode-tabs">
        <button className={`tab-btn ${mode === 'upload' ? 'active' : ''}`} onClick={() => switchMode('upload')}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
          Upload Image
        </button>

        <button className={`tab-btn ${mode === 'camera' ? 'active' : ''}`} onClick={() => switchMode('camera')}>
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
            <circle cx="12" cy="13" r="4" />
          </svg>
          Live Camera
          {isStreaming && <span className="live-dot-pulse" />}
        </button>
      </nav>

      <main className="main">
        {mode === 'upload' && (
          <>
            <section className="upload-section">
              <div
                className={`dropzone ${dragActive ? 'drag-active' : ''} ${preview ? 'has-file' : ''}`}
                onDragEnter={handleDrag}
                onDragLeave={handleDrag}
                onDragOver={handleDrag}
                onDrop={handleDrop}
                onClick={() => fileInputRef.current?.click()}
              >
                <input ref={fileInputRef} type="file" accept="image/*" onChange={handleFileInput} className="file-input" />
                {preview ? (
                  <div className="preview-container">
                    <img src={preview} alt="Selected" className="preview-image" />
                    <div className="preview-overlay"><span>Click or drag to change image</span></div>
                  </div>
                ) : (
                  <div className="dropzone-content">
                    <div className="upload-icon">
                      <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
                        <polyline points="17 8 12 3 7 8" />
                        <line x1="12" y1="3" x2="12" y2="15" />
                      </svg>
                    </div>
                    <p className="dropzone-text">Drag & drop a vehicle image here</p>
                    <p className="dropzone-subtext">or click to browse files</p>
                    <p className="dropzone-formats">Supports: JPG, PNG, WEBP, BMP</p>
                  </div>
                )}
              </div>

              <div className="action-buttons">
                <button className="btn btn-primary" onClick={handleSubmit} disabled={!selectedFile || loading}>
                  {loading ? (
                    <><span className="spinner" /> Analyzing & Reading Plate...</>
                  ) : (
                    <>
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="11" cy="11" r="8" />
                        <path d="m21 21-4.35-4.35" />
                      </svg>
                      Detect Plates & Read Text
                    </>
                  )}
                </button>
                {selectedFile && (
                  <button className="btn btn-secondary" onClick={resetAll} disabled={loading}>
                    Reset
                  </button>
                )}
              </div>
            </section>

            {error && (
              <div className="error-card">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="15" y1="9" x2="9" y2="15" />
                  <line x1="9" y1="9" x2="15" y2="15" />
                </svg>
                <span>{error}</span>
              </div>
            )}

            {loading && (
              <div className="loading-card">
                <div className="loading-spinner" />
                <p>Detecting number plate & extracting characters...</p>
                <p className="loading-subtext">Applying Deskewing + Bilateral CLAHE Enhancement + Multi-Line OCR</p>
              </div>
            )}

            {results && (
              <section className="results-section">
                <h2 className="section-title">
                  <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M22 11.08V12a10 10 0 1 1-5.93-9.14" />
                    <polyline points="22 4 12 14.01 9 11.01" />
                  </svg>
                  Detection & Recognition Results
                  <span className="plate-count">{results.plates?.length || 0} plate{results.plates?.length !== 1 ? 's' : ''} found</span>
                </h2>

                {results.annotated_image && (
                  <div className="result-card annotated-card">
                    <div className="card-header-bar">
                      <h3>Annotated Image (Bounding Box & Detected Text)</h3>
                    </div>
                    <div className="annotated-image-wrapper">
                      <img src={results.annotated_image} alt="Annotated result" className="annotated-image" />
                    </div>
                  </div>
                )}

                {results.plates && results.plates.length > 0 ? (
                  <div className="plates-grid">
                    {results.plates.map((plate, index) => (
                      <div key={index} className="plate-card">
                        <div className="plate-card-header">
                          <span className="plate-number">Vehicle #{index + 1}</span>
                          {plate.confidence != null && (
                            <span className="confidence-badge">{(plate.confidence * 100).toFixed(1)}% match</span>
                          )}
                        </div>

                        <div className="authentic-plate-container">
                          <span className="plate-label">Recognized Number Plate</span>
                          <div className="realistic-plate-banner">
                            <div className="plate-country-strip">
                              <span className="plate-emblem" />
                              <span className="plate-country-code">IND</span>
                            </div>
                            <div className="plate-text-embossed">
                              {plate.text && plate.text !== 'Plate Detected' ? plate.text : 'NO TEXT DETECTED'}
                            </div>
                            <button
                              className="plate-copy-btn"
                              title="Copy Plate Text"
                              onClick={() => copyToClipboard(plate.text, `upload-${index}`)}
                            >
                              {copiedId === `upload-${index}` ? (
                                <span className="copied-text">✓ Copied</span>
                              ) : (
                                <svg width="15" height="15" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                                  <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                                  <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
                                </svg>
                              )}
                            </button>
                          </div>
                        </div>

                        <div className="crops-row">
                          {results.plate_images && results.plate_images[index] && (
                            <div className="crop-box">
                              <span className="crop-label">Original Crop</span>
                              <div className="plate-image-wrapper">
                                <img src={results.plate_images[index]} alt={`Plate ${index + 1}`} className="plate-image" />
                              </div>
                            </div>
                          )}

                          {results.processed_images && results.processed_images[index] && (
                            <div className="crop-box">
                              <span className="crop-label">Enhanced IPCV Filter</span>
                              <div className="plate-image-wrapper">
                                <img src={results.processed_images[index]} alt={`Enhanced Plate ${index + 1}`} className="plate-image" />
                              </div>
                            </div>
                          )}
                        </div>

                        {plate.bbox && (
                          <div className="plate-bbox">
                            <span className="bbox-label">Bounding Box Coordinates</span>
                            <span className="bbox-value">[x: {plate.bbox[0]}, y: {plate.bbox[1]}, w: {plate.bbox[2]}, h: {plate.bbox[3]}]</span>
                          </div>
                        )}
                      </div>
                    ))}
                  </div>
                ) : (
                  <div className="no-plates-card">
                    <p>No license plates detected in this image.</p>
                    <p className="no-plates-hint">Try an image with a clearer, front-facing license plate.</p>
                  </div>
                )}
              </section>
            )}
          </>
        )}

        {mode === 'camera' && (
          <section className="camera-section">
            <div className="camera-toolbar">
              <div className="camera-controls">
                {!isStreaming ? (
                  <button className="btn btn-primary" onClick={startCamera}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polygon points="5 3 19 12 5 21 5 3" />
                    </svg>
                    Start Camera
                  </button>
                ) : (
                  <button className="btn btn-danger" onClick={stopCamera}>
                    <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <rect x="6" y="6" width="12" height="12" />
                    </svg>
                    Stop Camera
                  </button>
                )}

                {isStreaming && (
                  <>
                    <button
                      className={`btn ${isDetectingLive ? 'btn-success' : 'btn-secondary'}`}
                      onClick={() => setIsDetectingLive(!isDetectingLive)}
                    >
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <circle cx="12" cy="12" r="10" />
                        <line x1="2" y1="12" x2="22" y2="12" />
                        <path d="M12 2a15.3 15.3 0 0 1 4 10 15.3 15.3 0 0 1-4 10 15.3 15.3 0 0 1-4-10 15.3 15.3 0 0 1 4-10z" />
                      </svg>
                      {isDetectingLive ? 'Live Tracking: ON' : 'Live Tracking: PAUSED'}
                    </button>

                    <button className="btn btn-secondary" onClick={toggleFacingMode} title="Flip camera">
                      <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <path d="M21.5 2v6h-6M21.34 15.57a10 10 0 1 1-.57-8.38l5.67-5.67" />
                      </svg>
                      Flip Camera
                    </button>
                  </>
                )}
              </div>

              {isStreaming && (
                <div className="camera-status-badges">
                  <span className="badge badge-live"><span className="live-dot" /> LIVE STREAM</span>
                  <span className="badge badge-fps">{fps} FPS</span>
                  {lastDetectedTime && <span className="badge badge-detected">Hit: {lastDetectedTime}</span>}
                </div>
              )}
            </div>

            {cameraError && (
              <div className="error-card">
                <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                  <circle cx="12" cy="12" r="10" />
                  <line x1="15" y1="9" x2="9" y2="15" />
                  <line x1="9" y1="9" x2="15" y2="15" />
                </svg>
                <span>{cameraError}</span>
              </div>
            )}

            <div className="camera-viewport-card">
              <div className="camera-viewport">
                <video ref={videoRef} className="camera-video" autoPlay playsInline muted />
                <canvas ref={overlayCanvasRef} className="camera-overlay-canvas" />
                <canvas ref={captureCanvasRef} style={{ display: 'none' }} />

                {!isStreaming && (
                  <div className="camera-idle-placeholder" onClick={startCamera}>
                    <div className="camera-idle-icon">
                      <svg width="64" height="64" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5">
                        <path d="M23 19a2 2 0 0 1-2 2H3a2 2 0 0 1-2-2V8a2 2 0 0 1 2-2h4l2-3h6l2 3h4a2 2 0 0 1 2 2z" />
                        <circle cx="12" cy="13" r="4" />
                      </svg>
                    </div>
                    <h3>Camera Inactive</h3>
                    <p>Click "Start Camera" to detect & recognize vehicle number plates live</p>
                    <span className="btn btn-primary btn-sm">Turn On Camera</span>
                  </div>
                )}

                {isStreaming && isDetectingLive && <div className="scanning-laser-line" />}
              </div>
            </div>

            {livePlates.length > 0 && (
              <div className="live-feed-section">
                <h3 className="live-feed-title">
                  <span className="live-feed-pulse" />
                  Live Detected License Plates ({livePlates.length})
                </h3>

                <div className="live-plates-grid">
                  {livePlates.map((item) => (
                    <div key={item.id} className="live-plate-card">
                      {item.image && (
                        <div className="live-plate-crop">
                          <img src={item.image} alt="Plate Crop" />
                        </div>
                      )}

                      <div className="live-plate-info">
                        <div className="live-realistic-plate">
                          <div className="live-plate-strip">IND</div>
                          <span className="live-plate-text-embossed">
                            {item.text && item.text !== 'Plate Detected' ? item.text : 'DETECTING...'}
                          </span>
                        </div>

                        <div className="live-plate-meta">
                          <span className="confidence-badge">{(item.confidence * 100).toFixed(0)}% match</span>
                          <span className="live-plate-time">{item.time}</span>
                          <button
                            className="mini-copy-btn"
                            title="Copy Plate Number"
                            onClick={() => copyToClipboard(item.text, item.id)}
                          >
                            {copiedId === item.id ? 'Copied!' : 'Copy'}
                          </button>
                        </div>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </section>
        )}
      </main>

      <footer className="footer">
        <p>Number Plate Detection & Text Recognition — OpenCV Haar Cascade + EasyOCR</p>
      </footer>
    </div>
  );
}

export default App;
