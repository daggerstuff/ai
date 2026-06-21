import tempfile
from pathlib import Path

from training.scripts.filter_crisis_quality import filter_crisis_dataset


def test_filter_crisis_dataset_empty_input():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "empty_input.jsonl"
        output_file = Path(temp_dir) / "output.jsonl"

        # Create an empty input file
        input_file.touch()

        # Run the function with the empty file
        filter_crisis_dataset(str(input_file), str(output_file))

        # Check that the output file is created and is empty
        assert output_file.exists()
        assert output_file.stat().st_size == 0


def test_filter_crisis_dataset_invalid_json():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "invalid_input.jsonl"
        output_file = Path(temp_dir) / "output.jsonl"

        with open(input_file, "w") as f:
            f.write("this is not json\n")
            f.write("neither is this\n")

        filter_crisis_dataset(str(input_file), str(output_file))

        assert output_file.exists()
        assert output_file.stat().st_size == 0


def test_filter_crisis_dataset_empty_conversation():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "empty_conv.jsonl"
        output_file = Path(temp_dir) / "output.jsonl"

        import json

        with open(input_file, "w") as f:
            f.write(json.dumps({"conversation": []}) + "\n")

        filter_crisis_dataset(str(input_file), str(output_file))

        assert output_file.exists()
        assert output_file.stat().st_size == 0


def test_filter_crisis_dataset_keep_valid():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "valid.jsonl"
        output_file = Path(temp_dir) / "output.jsonl"

        import json

        with open(input_file, "w") as f:
            valid_conv = {
                "conversation": [
                    {"content": "I am feeling ok today."},
                    {"content": "That is good to hear."},
                ]
            }
            f.write(json.dumps(valid_conv) + "\n")

        filter_crisis_dataset(str(input_file), str(output_file))

        assert output_file.exists()
        with open(output_file) as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == valid_conv


def test_filter_crisis_dataset_remove_unaligned():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "unaligned.jsonl"
        output_file = Path(temp_dir) / "output.jsonl"

        import json

        with open(input_file, "w") as f:
            unaligned_conv = {
                "conversation": [
                    {"content": "I like ice cream."},
                    {
                        "content": "If you are considering suicide, please call for help."
                    },
                ]
            }
            f.write(json.dumps(unaligned_conv) + "\n")

        filter_crisis_dataset(str(input_file), str(output_file))

        assert output_file.exists()
        assert output_file.stat().st_size == 0


def test_filter_crisis_dataset_keep_aligned_crisis():
    with tempfile.TemporaryDirectory() as temp_dir:
        input_file = Path(temp_dir) / "aligned.jsonl"
        output_file = Path(temp_dir) / "output.jsonl"

        import json

        with open(input_file, "w") as f:
            aligned_conv = {
                "conversation": [
                    {"content": "I want to kill myself."},
                    {"content": "I can help you through this moment."},
                ]
            }
            f.write(json.dumps(aligned_conv) + "\n")

        filter_crisis_dataset(str(input_file), str(output_file))

        assert output_file.exists()
        with open(output_file) as f:
            lines = f.readlines()
        assert len(lines) == 1
        assert json.loads(lines[0]) == aligned_conv
