import React from 'react';
import { Link } from 'react-router-dom';
import './Navigation.css';

const Navigation = () => {
  return (
    <nav className="navigation">
      <div className="nav-brand">
        <Link to="/">CryptoDetect ERP</Link>
      </div>
      <ul className="nav-menu">
        <li><Link to="/">Dashboard</Link></li>
        <li><Link to="/upload">Upload</Link></li>
      </ul>
    </nav>
  );
};

export default Navigation;
