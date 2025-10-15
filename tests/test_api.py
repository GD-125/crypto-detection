"""
Comprehensive API Tests for Cryptographic Function Detection System
"""

import pytest
from fastapi.testclient import TestClient
from services.api.main import app

client = TestClient(app)


class TestRootEndpoints:
    """Test root API endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint returns correct information"""
        response = client.get("/")
        assert response.status_code == 200

        data = response.json()
        assert "message" in data
        assert data["message"] == "Cryptographic Function Detection API"
        assert "version" in data
        assert data["version"] == "1.0.0"
        assert "status" in data

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")
        assert response.status_code == 200

        data = response.json()
        assert data["status"] == "healthy"
        assert "timestamp" in data
        assert "services" in data

        services = data["services"]
        assert "api" in services
        assert services["api"] == "running"


class TestFirmwareEndpoints:
    """Test firmware management endpoints"""

    def test_firmware_list_empty(self):
        """Test listing firmware when empty"""
        response = client.get("/api/firmware/list")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_firmware_list_pagination(self):
        """Test firmware list with pagination"""
        response = client.get("/api/firmware/list?skip=0&limit=10")
        assert response.status_code == 200

        data = response.json()
        assert isinstance(data, list)
        assert len(data) <= 10

    def test_get_nonexistent_firmware(self):
        """Test getting firmware that doesn't exist"""
        response = client.get("/api/firmware/99999")
        assert response.status_code == 404


class TestAnalysisEndpoints:
    """Test analysis endpoints"""

    def test_analysis_status_not_found(self):
        """Test getting status for non-existent firmware"""
        response = client.get("/api/analysis/status/99999")
        assert response.status_code == 404

    def test_start_analysis_not_found(self):
        """Test starting analysis for non-existent firmware"""
        response = client.post("/api/analysis/start/99999")
        assert response.status_code == 404

    def test_batch_analysis_empty(self):
        """Test batch analysis with empty list"""
        response = client.post("/api/analysis/batch", json=[])
        assert response.status_code == 200

        data = response.json()
        assert data["total"] == 0
        assert data["started"] == 0


class TestResultsEndpoints:
    """Test results endpoints"""

    def test_list_all_results(self):
        """Test listing all results"""
        response = client.get("/api/results/list/all")
        assert response.status_code == 200
        assert isinstance(response.json(), list)

    def test_get_results_not_found(self):
        """Test getting results for non-existent firmware"""
        response = client.get("/api/results/99999")
        assert response.status_code == 404

    def test_export_results_not_found(self):
        """Test exporting results for non-existent firmware"""
        response = client.get("/api/results/export/99999?format=json")
        assert response.status_code == 404


class TestDashboardEndpoints:
    """Test dashboard endpoints"""

    def test_dashboard_stats(self):
        """Test dashboard statistics endpoint"""
        response = client.get("/api/dashboard/stats")
        assert response.status_code == 200

        data = response.json()
        assert "total_firmware" in data
        assert "analyzed_count" in data
        assert "analyzing_count" in data
        assert "pending_count" in data
        assert "total_crypto_functions" in data
        assert "timestamp" in data

        # All counts should be non-negative
        assert data["total_firmware"] >= 0
        assert data["analyzed_count"] >= 0
        assert data["analyzing_count"] >= 0

    def test_recent_activity(self):
        """Test recent activity endpoint"""
        response = client.get("/api/dashboard/recent-activity")
        assert response.status_code == 200

        data = response.json()
        assert "recent_uploads" in data
        assert "recent_analyses" in data
        assert "timestamp" in data

        assert isinstance(data["recent_uploads"], list)
        assert isinstance(data["recent_analyses"], list)

    def test_crypto_function_stats(self):
        """Test crypto function statistics endpoint"""
        response = client.get("/api/dashboard/crypto-functions")
        assert response.status_code == 200

        data = response.json()
        assert "function_types" in data
        assert "total_detections" in data
        assert "timestamp" in data


class TestAPIValidation:
    """Test API input validation"""

    def test_invalid_firmware_id(self):
        """Test invalid firmware ID type"""
        response = client.get("/api/firmware/invalid_id")
        assert response.status_code == 422

    def test_negative_pagination(self):
        """Test negative pagination values"""
        # Should handle gracefully
        response = client.get("/api/firmware/list?skip=-1&limit=-1")
        assert response.status_code in [200, 422]


class TestErrorHandling:
    """Test error handling"""

    def test_method_not_allowed(self):
        """Test method not allowed"""
        response = client.delete("/")
        assert response.status_code == 405

    def test_not_found(self):
        """Test 404 for non-existent route"""
        response = client.get("/api/nonexistent")
        assert response.status_code == 404
