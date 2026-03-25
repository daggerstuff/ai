import os
import unittest
from unittest.mock import patch

import pandas as pd

from ai.core.pipelines.processing.clean import clean_and_deduplicate


class TestCuratorVerification(unittest.TestCase):
    @patch("ai.pipelines.processing.clean.NemoCuratorClient")
    def test_curator_cleaning_call(self, mock_client_cls):
        """Verify clean_and_deduplicate calls NeMo Curator when enabled."""
        with patch.dict(os.environ, {"USE_NVIDIA_CURATOR": "true"}):
            df_input = pd.DataFrame({"text": ["I feel sad"], "other": ["data"]})

            mock_instance = mock_client_cls.return_value
            # Mock the curate_therapeutic_data call
            mock_instance.curate_therapeutic_data.return_value = {"status": "success"}

            df_output = clean_and_deduplicate(df_input)

            # Check if client was initialized and method called
            self.assertTrue(mock_client_cls.called)
            self.assertTrue(mock_instance.curate_therapeutic_data.called)
            self.assertEqual(len(df_output), 1)
            print("✅ NeMo Curator cleaning call verified")


if __name__ == "__main__":
    unittest.main()
