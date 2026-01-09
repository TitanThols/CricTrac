import React from "react";

const VideoPlayer = ({ videoUrl }) => {
  if (!videoUrl) return null;

  // Backend outputs:   "/outputs/processed_sample.mp4"
  // Full URL must be:  "http://localhost:8000/outputs/processed_sample.mp4"
  const fullUrl = "http://localhost:8000" + videoUrl;

  return (
    <div style={{ marginTop: "30px" }}>
      <h2>Processed Output</h2>

      <video
        src={fullUrl}
        controls
        style={{ width: "70%", borderRadius: "8px", border: "2px solid #222" }}
      />

      <br />
      <br />

      <a href={fullUrl} download>
        <button>Download Output</button>
      </a>
    </div>
  );
};

export default VideoPlayer;
