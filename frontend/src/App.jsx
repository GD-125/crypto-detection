import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import Dashboard from './components/Dashboard';
import FirmwareUpload from './components/FirmwareUpload';
import AnalysisResults from './components/AnalysisResults';
import Navigation from './components/Navigation';
import './App.css';

function App() {
  return (
    <Router>
      <div className="App">
        <Navigation />
        <div className="container">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/upload" element={<FirmwareUpload />} />
            <Route path="/results/:id" element={<AnalysisResults />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
