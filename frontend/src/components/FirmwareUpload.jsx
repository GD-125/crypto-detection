import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import apiService from '../services/api';
import './FirmwareUpload.css';

const FirmwareUpload = () => {
  const [file, setFile] = useState(null);
  const [architecture, setArchitecture] = useState('auto');
  const [uploading, setUploading] = useState(false);
  const [message, setMessage] = useState('');
  const navigate = useNavigate();

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setMessage('');
  };

  const handleUpload = async (e) => {
    e.preventDefault();

    if (!file) {
      setMessage('Please select a file');
      return;
    }

    setUploading(true);
    setMessage('');

    try {
      const response = await apiService.uploadFirmware(file, architecture);
      setMessage(`Upload successful! File: ${response.filename}`);

      // Start analysis automatically
      await apiService.startAnalysis(response.id);
      setMessage('Upload successful! Analysis started...');

      // Redirect to results page after a delay
      setTimeout(() => {
        navigate(`/results/${response.id}`);
      }, 2000);
    } catch (error) {
      setMessage(`Upload failed: ${error.message}`);
    } finally {
      setUploading(false);
    }
  };

  return (
    <div className="firmware-upload">
      <h1>Upload Firmware for Analysis</h1>

      <form onSubmit={handleUpload} className="upload-form">
        <div className="form-group">
          <label htmlFor="file">Select Firmware Binary:</label>
          <input
            type="file"
            id="file"
            onChange={handleFileChange}
            disabled={uploading}
            accept=".bin,.elf,.hex,.fw"
          />
        </div>

        <div className="form-group">
          <label htmlFor="architecture">Architecture:</label>
          <select
            id="architecture"
            value={architecture}
            onChange={(e) => setArchitecture(e.target.value)}
            disabled={uploading}
          >
            <option value="auto">Auto-detect</option>
            <option value="x86">x86</option>
            <option value="x86_64">x86_64</option>
            <option value="arm">ARM</option>
            <option value="arm64">ARM64</option>
            <option value="mips">MIPS</option>
            <option value="powerpc">PowerPC</option>
          </select>
        </div>

        <button
          type="submit"
          className="btn-primary"
          disabled={uploading || !file}
        >
          {uploading ? 'Uploading...' : 'Upload and Analyze'}
        </button>

        {message && (
          <div className={`message ${message.includes('failed') ? 'error' : 'success'}`}>
            {message}
          </div>
        )}
      </form>

      <div className="upload-info">
        <h3>Supported Formats:</h3>
        <ul>
          <li>ELF binaries (.elf)</li>
          <li>Raw binary files (.bin)</li>
          <li>Intel HEX files (.hex)</li>
          <li>Firmware images (.fw)</li>
        </ul>

        <h3>Analysis Process:</h3>
        <ol>
          <li>Binary disassembly using Ghidra</li>
          <li>Feature extraction from multiple ISAs</li>
          <li>AI-powered cryptographic function detection</li>
          <li>Explainable AI results with confidence scores</li>
        </ol>
      </div>
    </div>
  );
};

export default FirmwareUpload;
