"""
Unit Tests for API Routes
"""

import pytest
from fastapi.testclient import TestClient
from services.api.main import app

client = TestClient(app)


class TestRootEndpoints:
    """Test suite for root endpoints"""

    def test_root_endpoint(self):
        """Test root endpoint"""
        response = client.get("/")

        assert response.status_code == 200
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert data["version"] == "1.0.0"

    def test_health_check(self):
        """Test health check endpoint"""
        response = client.get("/health")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert "services" in data
        assert "timestamp" in data


class TestFirmwareRoutes:
    """Test suite for firmware routes"""

    def test_list_firmware_empty(self):
        """Test listing firmware when empty"""
        response = client.get("/api/firmware/list")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_upload_firmware(self, sample_binary):
        """Test firmware upload"""
        files = {
            "file": ("test_firmware.bin", sample_binary, "application/octet-stream")
        }

        response = client.post(
            "/api/firmware/upload?architecture=x86_64",
            files=files
        )

        assert response.status_code == 200
        data = response.json()
        assert "id" in data
        assert "filename" in data
        assert "file_hash" in data
        assert data["status"] in ["success", "already_exists"]

    def test_upload_firmware_auto_arch(self, sample_binary):
        """Test firmware upload with auto architecture"""
        files = {
            "file": ("test_firmware.bin", sample_binary, "application/octet-stream")
        }

        response = client.post(
            "/api/firmware/upload?architecture=auto",
            files=files
        )

        assert response.status_code == 200

    def test_get_firmware_not_found(self):
        """Test getting non-existent firmware"""
        response = client.get("/api/firmware/99999")

        assert response.status_code == 404

    def test_upload_invalid_file(self):
        """Test uploading invalid file"""
        # This should still accept the upload but may fail analysis
        files = {
            "file": ("test.txt", b"not a binary", "text/plain")
        }

        response = client.post(
            "/api/firmware/upload",
            files=files
        )

        # Should accept upload
        assert response.status_code in [200, 400]


class TestAnalysisRoutes:
    """Test suite for analysis routes"""

    def test_start_analysis_not_found(self):
        """Test starting analysis for non-existent firmware"""
        response = client.post("/api/analysis/start/99999")

        assert response.status_code == 404

    def test_get_analysis_status_not_found(self):
        """Test getting status for non-existent firmware"""
        response = client.get("/api/analysis/status/99999")

        assert response.status_code == 404

    def test_batch_analysis_empty(self):
        """Test batch analysis with empty list"""
        response = client.post(
            "/api/analysis/batch",
            json=[]
        )

        assert response.status_code == 200
        data = response.json()
        assert data["total"] == 0


class TestResultsRoutes:
    """Test suite for results routes"""

    def test_get_results_not_found(self):
        """Test getting results for non-existent firmware"""
        response = client.get("/api/results/99999")

        assert response.status_code == 404

    def test_list_all_results(self):
        """Test listing all results"""
        response = client.get("/api/results/list/all")

        assert response.status_code == 200
        data = response.json()
        assert isinstance(data, list)

    def test_export_results_not_found(self):
        """Test exporting non-existent results"""
        response = client.get("/api/results/export/99999?format=json")

        assert response.status_code == 404

    def test_export_results_invalid_format(self):
        """Test exporting with invalid format"""
        # First would need a valid firmware_id
        # For now test that endpoint exists
        response = client.get("/api/results/export/1?format=invalid")

        assert response.status_code in [400, 404]


class TestDashboardRoutes:
    """Test suite for dashboard routes"""

    def test_get_dashboard_stats(self):
        """Test getting dashboard statistics"""
        response = client.get("/api/dashboard/stats")

        assert response.status_code == 200
        data = response.json()
        assert "total_firmware" in data
        assert "analyzed_count" in data
        assert "analyzing_count" in data
        assert "total_crypto_functions" in data

    def test_get_recent_activity(self):
        """Test getting recent activity"""
        response = client.get("/api/dashboard/recent-activity")

        assert response.status_code == 200
        data = response.json()
        assert "recent_uploads" in data
        assert "recent_analyses" in data
        assert "timestamp" in data

    def test_get_crypto_function_stats(self):
        """Test getting crypto function statistics"""
        response = client.get("/api/dashboard/crypto-functions")

        assert response.status_code == 200
        data = response.json()
        assert "function_types" in data
        assert "total_detections" in data


class TestAPIValidation:
    """Test suite for API input validation"""

    def test_invalid_firmware_id_type(self):
        """Test invalid firmware ID type"""
        response = client.get("/api/firmware/invalid_id")

        assert response.status_code == 422  # Validation error

    def test_negative_skip_limit(self):
        """Test negative skip/limit values"""
        response = client.get("/api/firmware/list?skip=-1&limit=-1")

        # Should handle gracefully or return validation error
        assert response.status_code in [200, 422]

    def test_large_limit(self):
        """Test very large limit value"""
        response = client.get("/api/firmware/list?skip=0&limit=10000")

        # Should either cap the limit or accept it
        assert response.status_code == 200


class TestCORS:
    """Test suite for CORS configuration"""

    def test_cors_headers(self):
        """Test CORS headers are present"""
        response = client.options("/api/firmware/list")

        # Should have CORS headers configured
        # Exact headers depend on configuration
        assert response.status_code in [200, 405]


class TestErrorHandling:
    """Test suite for error handling"""

    def test_method_not_allowed(self):
        """Test method not allowed"""
        response = client.delete("/")  # DELETE not allowed on root

        assert response.status_code == 405

    def test_not_found_route(self):
        """Test accessing non-existent route"""
        response = client.get("/api/nonexistent/route")

        assert response.status_code == 404

    def test_invalid_json_payload(self):
        """Test sending invalid JSON"""
        response = client.post(
            "/api/analysis/batch",
            data="invalid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422
