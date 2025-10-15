import React, { useState, useEffect } from 'react';
import { useParams } from 'react-router-dom';
import apiService from '../services/api';
import './AnalysisResults.css';

const AnalysisResults = () => {
  const { id } = useParams();
  const [results, setResults] = useState(null);
  const [firmware, setFirmware] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetchResults();
    const interval = setInterval(checkStatus, 5000); // Poll every 5 seconds
    return () => clearInterval(interval);
  }, [id]);

  const fetchResults = async () => {
    try {
      const [firmwareData, resultsData] = await Promise.all([
        apiService.getFirmware(id),
        apiService.getResults(id)
      ]);
      setFirmware(firmwareData);
      setResults(resultsData);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching results:', error);
      setError(error.message);
      setLoading(false);
    }
  };

  const checkStatus = async () => {
    try {
      const status = await apiService.getAnalysisStatus(id);
      if (status.status === 'analyzed' && !results) {
        fetchResults();
      }
    } catch (error) {
      console.error('Error checking status:', error);
    }
  };

  if (loading) {
    return <div className="loading">Loading results...</div>;
  }

  if (error) {
    return <div className="error">Error: {error}</div>;
  }

  if (firmware?.status === 'analyzing') {
    return (
      <div className="analyzing">
        <h2>Analysis in Progress</h2>
        <p>Please wait while we analyze the firmware...</p>
        <div className="spinner"></div>
      </div>
    );
  }

  return (
    <div className="analysis-results">
      <h1>Analysis Results</h1>

      {/* Firmware Info */}
      <div className="firmware-info">
        <h2>Firmware Information</h2>
        <div className="info-grid">
          <div className="info-item">
            <strong>Filename:</strong> {firmware?.filename}
          </div>
          <div className="info-item">
            <strong>Architecture:</strong> {firmware?.architecture}
          </div>
          <div className="info-item">
            <strong>File Size:</strong> {(firmware?.file_size / 1024).toFixed(2)} KB
          </div>
          <div className="info-item">
            <strong>Upload Time:</strong> {new Date(firmware?.upload_time).toLocaleString()}
          </div>
        </div>
      </div>

      {/* Detected Functions */}
      <div className="detected-functions">
        <h2>Detected Cryptographic Functions</h2>
        {results?.detected_functions && results.detected_functions.length > 0 ? (
          <div className="functions-grid">
            {results.detected_functions.map((func, index) => (
              <div key={index} className="function-card">
                <h3>{func.type}</h3>
                <div className="confidence-bar">
                  <div
                    className="confidence-fill"
                    style={{ width: `${func.confidence * 100}%` }}
                  ></div>
                </div>
                <p className="confidence-text">
                  Confidence: {(func.confidence * 100).toFixed(2)}%
                </p>
                <span className="rank-badge">Rank #{func.rank}</span>
              </div>
            ))}
          </div>
        ) : (
          <p>No cryptographic functions detected</p>
        )}
      </div>

      {/* XAI Explanations */}
      <div className="xai-explanations">
        <h2>Explainable AI Analysis</h2>
        {results?.explanations && (
          <div className="explanation-content">
            <p><strong>Method:</strong> {results.explanations.method}</p>
            <p><strong>Summary:</strong> {results.explanations.summary}</p>

            <h3>Feature Importance</h3>
            <table className="feature-table">
              <thead>
                <tr>
                  <th>Feature Index</th>
                  <th>Importance</th>
                  <th>Contribution</th>
                </tr>
              </thead>
              <tbody>
                {results.explanations.feature_importance?.map((feature, index) => (
                  <tr key={index}>
                    <td>{feature.feature_index}</td>
                    <td>{feature.importance.toFixed(4)}</td>
                    <td>
                      <span className={`contribution ${feature.contribution}`}>
                        {feature.contribution}
                      </span>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>

      {/* Metadata */}
      <div className="metadata">
        <h2>Analysis Metadata</h2>
        <pre>{JSON.stringify(results?.metadata, null, 2)}</pre>
      </div>
    </div>
  );
};

export default AnalysisResults;
