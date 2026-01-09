import React, { useState } from "react";
import VideoUploader from "../components/VideoUploader";
import VideoPlayer from "../components/VideoPlayer";

const TrackNonML = () => {
  const [outputVideo, setOutputVideo] = useState(null);

  return (
    <div style={{ padding: "30px" }}>
      <h1>Non-ML Tracking</h1>

      <VideoUploader onComplete={setOutputVideo} />

      {outputVideo && <VideoPlayer videoUrl={outputVideo} />}
    </div>
  );
};

export default TrackNonML;
