import React, { useState } from "react";
import API from "../api/backend";

const VideoUploader = ({ onComplete }) => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);

  const uploadVideo = async () => {
    if (!file) {
      alert("Please select a video.");
      return;
    }

    const formData = new FormData();
    formData.append("video", file);

    setLoading(true);

    try {
      const res = await API.post("/track/non-ml", formData, {
        headers: { "Content-Type": "multipart/form-data" },
      });

      if (res.data.status === "success") {
        onComplete(res.data.output_video);
      } else {
        alert("Pipeline error: " + res.data.stderr);
      }
    } catch (err) {
      alert("Upload failed.");
    }

    setLoading(false);
  };

  return (
    <div style={{ marginTop: "20px" }}>
      <input
        type="file"
        accept="video/*"
        onChange={(e) => setFile(e.target.files[0])}
      />

      <button onClick={uploadVideo} style={{ marginLeft: "15px" }}>
        {loading ? "Processing..." : "Upload & Process"}
      </button>
    </div>
  );
};

export default VideoUploader;
