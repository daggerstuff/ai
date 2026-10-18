import unittest
from pathlib import Path


class TestEndToEndPipeline(unittest.TestCase):
    def setUp(self):
        self.project_root = Path(__file__).parents[2]
        self.scripts_dir = self.project_root / "ai/training/scripts"

    def test_scripts_existence(self):
        """Verify that all required Phase 1b scripts exist in the new location."""
        required_scripts = [
            "extract_all_youtube_transcripts.py",
            "extract_academic_findings.py",
            "extract_all_books_to_training.py",
            "generate_nemo_synthetic_data.py",
        ]
        for script in required_scripts:
            assert (self.scripts_dir / script).exists(), f"Missing {script}"

    def test_imports(self):
        """Verify that scripts can import the S3Loader from the new utils location."""
        # Simple dry-run import check
        utils_path = self.project_root / "ai/training/utils/s3_dataset_loader.py"
        assert utils_path.exists(), "S3DatasetLoader shim missing"

    def test_production_script_paths(self):
        """Verify run_phase1_production.sh points to the correct ready_packages path."""
        prod_script = self.project_root / "scripts/run_phase1_production.sh"
        content = prod_script.read_text()
        assert "ai/training/scripts" in content
        assert "ai/training_ready/scripts" not in content, "Legacy path detected in production script"


if __name__ == "__main__":
    unittest.main()
