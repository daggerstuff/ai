import pytest
from unittest.mock import patch
from training.configs.stage_configs.cli_config import get_config_value

def test_get_config_value_dot_notation():
    test_config = {
        "level1": {
            "level2": {
                "level3": "value3",
                "list_val": [1, 2, 3]
            },
            "other2": "value2"
        },
        "flat": "flat_value"
    }

    with patch("training.configs.stage_configs.cli_config._config_manager.load", return_value=test_config):
        assert get_config_value("flat") == "flat_value"
        assert get_config_value("level1.other2") == "value2"
        assert get_config_value("level1.level2.level3") == "value3"

        assert get_config_value("missing", "default") == "default"
        assert get_config_value("level1.missing", "default") == "default"
        assert get_config_value("level1.level2.missing", "default") == "default"
        assert get_config_value("flat.missing", "default") == "default"
        assert get_config_value("level1.level2.list_val.0", "default") == "default"
