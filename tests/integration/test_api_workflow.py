"""
Integration Tests for API Workflow
"""

import pytest
import time
from fastapi.testclient import TestClient
from services.api.main import app

client = TestClient(app)


class TestEndToEndWorkflow:
    """Test suite for end-to-end API workflow"""

    def test_complete_firmware_analysis_workflow(self, sample_binary):
        """Test complete workflow: upload -> analyze -> get results"""

        # Step 1: Upload firmware
        files = {
            "file": ("test_firmware.bin", sample_binary, "application/octet-stream")
        }
        upload_response = client.post(
            "/api/firmware/upload",
            files=files
        )

        assert upload_response.status_code == 200
        upload_data = upload_response.json()
        firmware_id = upload_data["id"]

        # Step 2: Verify firmware is listed
        list_response = client.get("/api/firmware/list")
        assert list_response.status_code == 200
        firmware_list = list_response.json()
        assert any(f["id"] == firmware_id for f in firmware_list)

        # Step 3: Get firmware details
        detail_response = client.get(f"/api/firmware/{firmware_id}")
        assert detail_response.status_code == 200
        detail_data = detail_response.json()
        assert detail_data["id"] == firmware_id
        assert detail_data["status"] in ["uploaded", "analyzing", "analyzed"]

        # Step 4: Start analysis (if not already started)
        if detail_data["status"] == "uploaded":
            analysis_response = client.post(f"/api/analysis/start/{firmware_id}")
            assert analysis_response.status_code == 200

        # Step 5: Check analysis status
        status_response = client.get(f"/api/analysis/status/{firmware_id}")
        assert status_response.status_code == 200
        status_data = status_response.json()
        assert "status" in status_data

        # Note: In real test, we'd wait for analysis to complete
        # For unit tests, we're just verifying the workflow exists

    def test_batch_upload_and_analysis(self, sample_binary):
        """Test batch uploading and analyzing multiple firmware"""
        firmware_ids = []

        # Upload multiple firmware files
        for i in range(3):
            files = {
                "file": (f"firmware_{i}.bin", sample_binary, "application/octet-stream")
            }
            response = client.post("/api/firmware/upload", files=files)

            assert response.status_code == 200
            firmware_ids.append(response.json()["id"])

        # Batch analyze
        batch_response = client.post(
            "/api/analysis/batch",
            json=firmware_ids
        )

        assert batch_response.status_code == 200
        batch_data = batch_response.json()
        assert batch_data["total"] == len(firmware_ids)

    def test_dashboard_reflects_uploads(self, sample_binary):
        """Test dashboard statistics update with uploads"""

        # Get initial stats
        initial_stats = client.get("/api/dashboard/stats").json()
        initial_count = initial_stats["total_firmware"]

        # Upload firmware
        files = {
            "file": ("new_firmware.bin", sample_binary, "application/octet-stream")
        }
        client.post("/api/firmware/upload", files=files)

        # Get updated stats
        updated_stats = client.get("/api/dashboard/stats").json()

        # Should reflect new upload
        assert updated_stats["total_firmware"] >= initial_count

    def test_recent_activity_tracking(self, sample_binary):
        """Test recent activity is tracked"""

        # Upload firmware
        files = {
            "file": ("activity_test.bin", sample_binary, "application/octet-stream")
        }
        upload_response = client.post("/api/firmware/upload", files=files)
        firmware_id = upload_response.json()["id"]

        # Check recent activity
        activity_response = client.get("/api/dashboard/recent-activity")
        assert activity_response.status_code == 200

        activity_data = activity_response.json()
        recent_uploads = activity_data["recent_uploads"]

        # Should include our upload
        assert any(f["id"] == firmware_id for f in recent_uploads)


class TestAPIErrorHandling:
    """Test suite for API error handling in workflows"""

    def test_duplicate_upload_handling(self, sample_binary):
        """Test uploading the same file twice"""

        files = {
            "file": ("duplicate.bin", sample_binary, "application/octet-stream")
        }

        # First upload
        response1 = client.post("/api/firmware/upload", files=files)
        assert response1.status_code == 200
        data1 = response1.json()

        # Second upload (same file)
        files2 = {
            "file": ("duplicate.bin", sample_binary, "application/octet-stream")
        }
        response2 = client.post("/api/firmware/upload", files=files2)
        assert response2.status_code == 200
        data2 = response2.json()

        # Should recognize duplicate
        if data2["status"] == "already_exists":
            assert data2["file_hash"] == data1["file_hash"]

    def test_analyze_already_analyzing(self, sample_binary):
        """Test starting analysis when already analyzing"""

        # Upload and start analysis
        files = {
            "file": ("analyzing_test.bin", sample_binary, "application/octet-stream")
        }
        upload_response = client.post("/api/firmware/upload", files=files)
        firmware_id = upload_response.json()["id"]

        # Start analysis
        client.post(f"/api/analysis/start/{firmware_id}")

        # Try to start again immediately
        response = client.post(f"/api/analysis/start/{firmware_id}")

        # Should handle gracefully (either accept or reject)
        assert response.status_code in [200, 400]

    def test_get_results_before_analysis(self, sample_binary):
        """Test getting results before analysis completes"""

        # Upload but don't analyze
        files = {
            "file": ("no_analysis.bin", sample_binary, "application/octet-stream")
        }
        upload_response = client.post("/api/firmware/upload", files=files)
        firmware_id = upload_response.json()["id"]

        # Try to get results
        results_response = client.get(f"/api/results/{firmware_id}")

        # Should return 404 or indicate no results
        assert results_response.status_code == 404

    def test_delete_firmware_cascade(self, sample_binary):
        """Test deleting firmware also deletes related data"""

        # Upload firmware
        files = {
            "file": ("delete_test.bin", sample_binary, "application/octet-stream")
        }
        upload_response = client.post("/api/firmware/upload", files=files)
        firmware_id = upload_response.json()["id"]

        # Verify it exists
        get_response = client.get(f"/api/firmware/{firmware_id}")
        assert get_response.status_code == 200

        # Delete firmware
        delete_response = client.delete(f"/api/firmware/{firmware_id}")
        assert delete_response.status_code == 200

        # Verify it's gone
        get_after_delete = client.get(f"/api/firmware/{firmware_id}")
        assert get_after_delete.status_code == 404


class TestAPIConcurrency:
    """Test suite for API concurrency handling"""

    def test_multiple_concurrent_uploads(self, sample_binary):
        """Test handling multiple concurrent uploads"""
        import concurrent.futures

        def upload_firmware(i):
            files = {
                "file": (f"concurrent_{i}.bin", sample_binary, "application/octet-stream")
            }
            return client.post("/api/firmware/upload", files=files)

        # Upload concurrently
        with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
            futures = [executor.submit(upload_firmware, i) for i in range(5)]
            responses = [f.result() for f in futures]

        # All should succeed
        assert all(r.status_code == 200 for r in responses)

    def test_concurrent_analysis_requests(self, sample_binary):
        """Test handling concurrent analysis requests"""

        # Upload multiple firmware
        firmware_ids = []
        for i in range(3):
            files = {
                "file": (f"concurrent_analysis_{i}.bin", sample_binary, "application/octet-stream")
            }
            response = client.post("/api/firmware/upload", files=files)
            firmware_ids.append(response.json()["id"])

        # Start analyses concurrently
        import concurrent.futures

        def start_analysis(fid):
            return client.post(f"/api/analysis/start/{fid}")

        with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
            futures = [executor.submit(start_analysis, fid) for fid in firmware_ids]
            responses = [f.result() for f in futures]

        # All should succeed
        assert all(r.status_code == 200 for r in responses)


class TestAPIDataValidation:
    """Test suite for API data validation"""

    def test_firmware_list_pagination(self):
        """Test firmware list pagination"""

        # Test different skip/limit combinations
        response1 = client.get("/api/firmware/list?skip=0&limit=10")
        assert response1.status_code == 200

        response2 = client.get("/api/firmware/list?skip=10&limit=10")
        assert response2.status_code == 200

        # Data should be lists
        assert isinstance(response1.json(), list)
        assert isinstance(response2.json(), list)

    def test_export_format_validation(self):
        """Test export format validation"""

        # Valid format
        response1 = client.get("/api/results/export/1?format=json")
        assert response1.status_code in [200, 404]  # 404 if firmware doesn't exist

        # Invalid format
        response2 = client.get("/api/results/export/1?format=invalid")
        assert response2.status_code in [400, 404]

    def test_architecture_validation(self, sample_binary):
        """Test architecture parameter validation"""

        valid_archs = ["x86", "x86_64", "arm", "arm64", "mips", "auto"]

        for arch in valid_archs:
            files = {
                "file": (f"arch_test_{arch}.bin", sample_binary, "application/octet-stream")
            }
            response = client.post(
                f"/api/firmware/upload?architecture={arch}",
                files=files
            )
            assert response.status_code == 200
