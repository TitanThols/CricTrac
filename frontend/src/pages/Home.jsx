import React, { useState, useRef } from "react";
import "./Home.css";
import {
  Upload,
  Activity,
  Zap,
  Check,
  AlertCircle,
  Download,
  PlayCircle,
  Camera,
  StopCircle,
} from "lucide-react";

export default function CricTrac() {
  const [file, setFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("ml");
  const [dragActive, setDragActive] = useState(false);

  // Webcam states
  const [stream, setStream] = useState(null);
  const [mediaRecorder, setMediaRecorder] = useState(null);
  const [recording, setRecording] = useState(false);
  const recordedChunks = useRef([]);

  /* ---------------- Drag & Drop ---------------- */
  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      stopWebcam();
      setFile(e.dataTransfer.files[0]);
      setError(null);
      setResult(null);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      stopWebcam();
      setFile(e.target.files[0]);
      setError(null);
      setResult(null);
    }
  };

  /* ---------------- Webcam ---------------- */
  const startWebcam = async () => {
    try {
      stopWebcam();
      const mediaStream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
      });

      const recorder = new MediaRecorder(mediaStream, {
        mimeType: "video/webm",
      });

      recordedChunks.current = [];

      recorder.ondataavailable = (e) => {
        if (e.data.size > 0) recordedChunks.current.push(e.data);
      };

      recorder.onstop = () => {
        const blob = new Blob(recordedChunks.current, {
          type: "video/webm",
        });

        const recordedFile = new File([blob], "webcam-recording.webm", {
          type: "video/webm",
        });

        setFile(recordedFile);
        recordedChunks.current = [];
      };

      setStream(mediaStream);
      setMediaRecorder(recorder);
      setError(null);
      setResult(null);
    } catch {
      setError("Webcam access denied");
    }
  };

  const startRecording = () => {
    if (!mediaRecorder) return;
    mediaRecorder.start();
    setRecording(true);
  };

  const stopRecording = () => {
    if (!mediaRecorder) return;
    mediaRecorder.stop();
    stopWebcam();
    setRecording(false);
  };

  const stopWebcam = () => {
    if (stream) {
      stream.getTracks().forEach((t) => t.stop());
      setStream(null);
      setMediaRecorder(null);
      setRecording(false);
    }
  };

  /* ---------------- Upload ---------------- */
  const handleUpload = async () => {
    if (!file) {
      setError("Please select or record a video");
      return;
    }

    const formData = new FormData();
    formData.append("video", file);

    setProcessing(true);
    setError(null);
    setResult(null);

    try {
      const res = await fetch(`http://localhost:8000/track/${mode}`, {
        method: "POST",
        body: formData,
      });

      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || "Processing failed");
      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="crictrac-container">
      <div className="content-wrapper">
        {/* Header */}
        <div className="header">
          <div className="logo-container">
            <Activity className="logo-icon" />
            <h1 className="logo-text">
              BTP<span className="logo-highlight">Project</span>
            </h1>
          </div>
          <p className="subtitle">Cricket Bat Tracking System</p>
        </div>

        {/* Mode Selector */}
        <div className="mode-selector">
          <button
            onClick={() => setMode("ml")}
            className={`mode-btn ${mode === "ml" ? "mode-btn-active" : ""}`}
          >
            <Zap /> ML Model
          </button>
          <button
            onClick={() => setMode("non-ml")}
            className={`mode-btn ${mode === "non-ml" ? "mode-btn-active" : ""}`}
          >
            <Activity /> Non-ML Pipeline
          </button>
        </div>

        {/* Webcam Preview */}
        {stream && (
          <video
            autoPlay
            muted
            playsInline
            className="webcam-preview"
            ref={(v) => v && (v.srcObject = stream)}
          />
        )}

        {/* Webcam Controls */}
        <div className="webcam-controls">
          {!stream && (
            <button onClick={startWebcam} className="process-btn">
              <Camera /> Open Webcam
            </button>
          )}
          {stream && !recording && (
            <button
              onClick={startRecording}
              className="process-btn process-btn-ml"
            >
              <PlayCircle /> Start Recording
            </button>
          )}
          {recording && (
            <button
              onClick={stopRecording}
              className="process-btn process-btn-nonml"
            >
              <StopCircle /> Stop Recording
            </button>
          )}
        </div>

        {/* Upload Box */}
        <div
          className={`upload-box ${dragActive ? "upload-box-active" : ""}`}
          onDragEnter={handleDrag}
          onDragLeave={handleDrag}
          onDragOver={handleDrag}
          onDrop={handleDrop}
        >
          <input
            type="file"
            accept="video/*"
            onChange={handleFileChange}
            className="file-input"
            id="file-upload"
          />
          <label htmlFor="file-upload" className="upload-label">
            <Upload />
            {file ? file.name : "Drop video or click to upload"}
          </label>
        </div>

        {/* Upload Button */}
        <button
          onClick={handleUpload}
          disabled={!file || processing}
          className="process-btn"
        >
          {processing ? "Processing..." : "Start Tracking"}
        </button>

        {/* Error */}
        {error && (
          <div className="error-container">
            <AlertCircle /> {error}
          </div>
        )}

        {/* Result */}
        {result && (
          <div className="result-container">
            <video
              controls
              src={`http://localhost:8000${result.output_video}`}
            />
            <a
              href={`http://localhost:8000${result.output_video}`}
              download
              className="download-btn"
            >
              <Download /> Download
            </a>
          </div>
        )}
      </div>
    </div>
  );
}
