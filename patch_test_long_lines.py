import re

with open("infra/cloud/distributed/test_checkpoint_system.py", "r") as f:
    content = f.read()

# Fix long lines in test_save_checkpoint
content = content.replace(
    "        loaded_metadata, loaded_data = self.manager.storage.load_checkpoint(metadata.checkpoint_id)",
    "        loaded_metadata, loaded_data = self.manager.storage.load_checkpoint(\n            metadata.checkpoint_id\n        )"
)

content = content.replace(
    "            assert not os.path.exists(failed_file_path), \"Failed checkpoint file was not cleaned up\"",
    "            assert not os.path.exists(\n                failed_file_path\n            ), \"Failed checkpoint file was not cleaned up\""
)

content = content.replace(
    "                \"details\": \"Successfully tested explicit save_checkpoint and error cleanup\",",
    "                \"details\": (\n                    \"Successfully tested explicit save_checkpoint and error cleanup\"\n                ),"
)

with open("infra/cloud/distributed/test_checkpoint_system.py", "w") as f:
    f.write(content)
