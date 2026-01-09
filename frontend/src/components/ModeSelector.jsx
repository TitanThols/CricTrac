import React from "react";
import { Link } from "react-router-dom";

const ModeSelector = () => {
  return (
    <div style={{ marginTop: "40px" }}>
      <h2>Select Mode</h2>

      <div style={{ marginTop: "20px" }}>
        <Link to="/non-ml">
          <button style={{ marginRight: "20px" }}>Non-ML Tracking</button>
        </Link>

        <Link to="/ml">
          <button>ML Tracking</button>
        </Link>
      </div>
    </div>
  );
};

export default ModeSelector;
