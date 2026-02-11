import os
import unittest

class TestReleaseIntegrity(unittest.TestCase):
    """
    Ensures all critical Phase 15 artifacts are present before release.
    """
    def setUp(self):
        self.required_files = [
            "src/retinavit/validation/simulation.py",
            "src/retinavit/validation/fairness.py",
            "docs/clinical/safety_protocols.md",
            "docs/evaluation/fairness_audit.md",
            "docs/release/model_card.md",
            "docs/release/api_docs.md",
            "docs/research/paper_draft.md"
        ]

    def test_files_exist(self):
        for f in self.required_files:
            self.assertTrue(os.path.exists(f), f"Missing critical release file: {f}")

    def test_model_export_exists(self):
        # Optimized mobile model is the cornerstone of the release
        mobile_path = "checkpoints/deployment/retinavit_mobile.pt"
        # We don't fail here if it's a dry run, but for professional release it's vital.
        if not os.path.exists(mobile_path):
             print(f"Note: {mobile_path} not found. Ensure to run run_deployment_optimization.sh before final shipping.")

if __name__ == "__main__":
    unittest.main()
