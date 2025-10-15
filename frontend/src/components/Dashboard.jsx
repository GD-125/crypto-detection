import React, { useState, useEffect } from 'react';
import { Link } from 'react-router-dom';
import apiService from '../services/api';
import './Dashboard.css';

const Dashboard = () => {
  const [stats, setStats] = useState(null);
  const [recentActivity, setRecentActivity] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    fetchDashboardData();
  }, []);

  const fetchDashboardData = async () => {
    try {
      const [statsData, activityData] = await Promise.all([
        apiService.getDashboardStats(),
        apiService.getRecentActivity()
      ]);
      setStats(statsData);
      setRecentActivity(activityData);
      setLoading(false);
    } catch (error) {
      console.error('Error fetching dashboard data:', error);
      setLoading(false);
    }
  };

  if (loading) {
    return <div className="loading">Loading dashboard...</div>;
  }

  return (
    <div className="dashboard">
      <h1>Cryptographic Primitives Detection System</h1>

      {/* Statistics Cards */}
      <div className="stats-grid">
        <div className="stat-card">
          <h3>Total Firmware</h3>
          <p className="stat-value">{stats?.total_firmware || 0}</p>
        </div>
        <div className="stat-card">
          <h3>Analyzed</h3>
          <p className="stat-value">{stats?.analyzed_count || 0}</p>
        </div>
        <div className="stat-card">
          <h3>In Progress</h3>
          <p className="stat-value">{stats?.analyzing_count || 0}</p>
        </div>
        <div className="stat-card">
          <h3>Crypto Functions</h3>
          <p className="stat-value">{stats?.total_crypto_functions || 0}</p>
        </div>
      </div>

      {/* Recent Activity */}
      <div className="recent-activity">
        <h2>Recent Uploads</h2>
        <table className="activity-table">
          <thead>
            <tr>
              <th>Filename</th>
              <th>Status</th>
              <th>Upload Time</th>
              <th>Actions</th>
            </tr>
          </thead>
          <tbody>
            {recentActivity?.recent_uploads?.map((firmware) => (
              <tr key={firmware.id}>
                <td>{firmware.filename}</td>
                <td>
                  <span className={`status-badge status-${firmware.status}`}>
                    {firmware.status}
                  </span>
                </td>
                <td>{new Date(firmware.upload_time).toLocaleString()}</td>
                <td>
                  <Link to={`/results/${firmware.id}`}>
                    <button className="btn-small">View</button>
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      {/* Quick Actions */}
      <div className="quick-actions">
        <Link to="/upload">
          <button className="btn-primary">Upload New Firmware</button>
        </Link>
      </div>
    </div>
  );
};

export default Dashboard;
