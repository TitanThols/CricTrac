import React, { useState } from "react";
import "./Home.css";
import {
  Upload,
  Activity,
  Zap,
  Check,
  AlertCircle,
  Download,
  PlayCircle,
} from "lucide-react";

export default function CricTrac() {
  const [file, setFile] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [mode, setMode] = useState("ml");
  const [dragActive, setDragActive] = useState(false);

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
      setFile(e.dataTransfer.files[0]);
      setError(null);
      setResult(null);
    }
  };

  const handleFileChange = (e) => {
    if (e.target.files && e.target.files[0]) {
      setFile(e.target.files[0]);
      setError(null);
      setResult(null);
    }
  };

  const handleUpload = async () => {
    if (!file) {
      setError("Please select a video file");
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

      if (!res.ok) {
        throw new Error(data.detail || "Processing failed");
      }

      setResult(data);
    } catch (err) {
      setError(err.message);
    } finally {
      setProcessing(false);
    }
  };

  return (
    <div className="crictrac-container">
      <div className="background-effects">
        <div className="bg-blob bg-blob-1"></div>
        <div className="bg-blob bg-blob-2"></div>
      </div>

      <div className="content-wrapper">
        {/* Header */}
        <div className="header">
          <div className="logo-container">
            <Activity className="logo-icon" />
            <h1 className="logo-text">
              Cric<span className="logo-highlight">Trac</span>
            </h1>
          </div>
          <p className="subtitle">Advanced Cricket Bat Tracking System</p>
        </div>

        {/* Mode Selection */}
        <div className="mode-selector-container">
          <div className="mode-selector">
            <button
              onClick={() => setMode("ml")}
              className={`mode-btn ${mode === "ml" ? "mode-btn-active mode-btn-ml" : ""}`}
            >
              <Zap className="mode-icon" />
              ML Pipeline
            </button>
            <button
              onClick={() => setMode("non-ml")}
              className={`mode-btn ${mode === "non-ml" ? "mode-btn-active mode-btn-nonml" : ""}`}
            >
              <Activity className="mode-icon" />
              Non-ML Pipeline
            </button>
          </div>
        </div>

        {/* Upload Section */}
        <div className="upload-container">
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
              <div className="upload-content">
                <div className="upload-icon-container">
                  <Upload className="upload-icon" />
                </div>

                {file ? (
                  <div className="file-info">
                    <p className="file-name">{file.name}</p>
                    <p className="file-size">
                      {(file.size / 1024 / 1024).toFixed(2)} MB
                    </p>
                  </div>
                ) : (
                  <div className="upload-text">
                    <p className="upload-title">Drop your video here</p>
                    <p className="upload-subtitle">or click to browse</p>
                    <p className="upload-formats">
                      Supports MP4, AVI, MOV, MKV
                    </p>
                  </div>
                )}
              </div>
            </label>
          </div>

          {/* Process Button */}
          <button
            onClick={handleUpload}
            disabled={!file || processing}
            className={`process-btn ${!file || processing ? "process-btn-disabled" : mode === "ml" ? "process-btn-ml" : "process-btn-nonml"}`}
          >
            {processing ? (
              <>
                <div className="spinner"></div>
                Processing Video...
              </>
            ) : (
              <>
                <PlayCircle className="process-icon" />
                Start Tracking
              </>
            )}
          </button>
        </div>

        {/* Error Display */}
        {error && (
          <div className="error-container">
            <AlertCircle className="error-icon" />
            <div>
              <p className="error-title">Processing Failed</p>
              <p className="error-message">{error}</p>
            </div>
          </div>
        )}

        {/* Success Result */}
        {result && (
          <div className="result-container">
            <div className="result-card">
              <div className="result-header">
                <Check className="result-check-icon" />
                <p className="result-title">Processing Complete!</p>
              </div>

              <div className="result-content">
                <video
                  controls
                  className="result-video"
                  src={`http://localhost:8000${result.output_video}`}
                >
                  Your browser doesn't support video playback.
                </video>

                <a
                  href={`http://localhost:8000${result.output_video}`}
                  download
                  className="download-btn"
                >
                  <Download className="download-icon" />
                  Download Processed Video
                </a>
              </div>
            </div>
          </div>
        )}

        {/* Footer */}
        <div className="footer">
          <p>Powered by Advanced Computer Vision & Machine Learning</p>
        </div>
      </div>
    </div>
  );
}
