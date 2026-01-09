import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";

import Home from "./pages/Home.jsx";
import TrackNonML from "./pages/TrackNonML.jsx";
import TrackML from "./pages/TrackMl.jsx";

function App() {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<Home />} />
        <Route path="/non-ml" element={<TrackNonML />} />
        <Route path="/ml" element={<TrackML />} />
      </Routes>
    </Router>
  );
}

export default App;
