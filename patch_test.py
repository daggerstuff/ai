import re

with open("infra/cloud/distributed/test_checkpoint_system.py", "r") as f:
    content = f.read()

test_method = """
    async def test_save_checkpoint(self):
        \"\"\"Explicitly test the save_checkpoint function including edge cases\"\"\"
        process_id = "test_save_chkpt_001"
        task_id = "save_test"

        test_data = {
            "key1": "value1",
            "key2": 42,
            "key3": [1, 2, 3]
        }

        # Create checkpoint metadata directly
        from checkpoint_system import CheckpointMetadata, CheckpointType, CheckpointStatus
        from datetime import datetime
        import uuid

        metadata = CheckpointMetadata(
            checkpoint_id=f"{process_id}_{task_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            checkpoint_type=CheckpointType.CUSTOM,
            created_at=datetime.utcnow(),
            process_id=process_id,
            task_id=task_id,
            description="Explicit save_checkpoint test",
            status=CheckpointStatus.ACTIVE,
            compression=False
        )

        # Test normal save_checkpoint
        file_path = self.manager.storage.save_checkpoint(metadata, test_data)

        # Verify file exists
        assert os.path.exists(file_path), "Checkpoint file was not created"

        # Verify data using load_checkpoint
        loaded_metadata, loaded_data = self.manager.storage.load_checkpoint(metadata.checkpoint_id)
        assert loaded_data == test_data, "Loaded data does not match saved data"
        assert loaded_metadata.checkpoint_id == metadata.checkpoint_id

        # Test error handling (e.g., trying to save unpickleable object)
        class Unpickleable:
            def __init__(self):
                self.lambda_func = lambda x: x

        unpickleable_data = Unpickleable()
        metadata_unpick = CheckpointMetadata(
            checkpoint_id=f"{process_id}_{task_id}_{int(time.time())}_{uuid.uuid4().hex[:8]}",
            checkpoint_type=CheckpointType.CUSTOM,
            created_at=datetime.utcnow(),
            process_id=process_id,
            task_id=task_id,
        )

        try:
            self.manager.storage.save_checkpoint(metadata_unpick, unpickleable_data)
            assert False, "Should have raised an exception for unpickleable data"
        except Exception as e:
            # Check if file was cleaned up on failure
            file_name = f"{metadata_unpick.checkpoint_id}.pkl"
            if metadata_unpick.compression:
                file_name += ".gz"
            failed_file_path = self.manager.storage.data_path / file_name
            assert not os.path.exists(failed_file_path), "Failed checkpoint file was not cleaned up"

        self.test_results.append(
            {
                "test": "test_save_checkpoint",
                "status": "passed",
                "details": "Successfully tested explicit save_checkpoint and error cleanup",
            }
        )
"""

if "async def test_save_checkpoint(self):" not in content:
    # Insert before print_test_summary
    content = content.replace("    def print_test_summary(self):", test_method + "\n    def print_test_summary(self):")
    with open("infra/cloud/distributed/test_checkpoint_system.py", "w") as f:
        f.write(content)
