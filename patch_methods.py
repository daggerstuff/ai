with open("infra/cloud/distributed/test_checkpoint_system.py", "r") as f:
    content = f.read()

if "self.test_save_checkpoint" not in content:
    content = content.replace(
        "self.test_compression_and_deduplication,\n        ]",
        "self.test_compression_and_deduplication,\n            self.test_save_checkpoint,\n        ]"
    )
    with open("infra/cloud/distributed/test_checkpoint_system.py", "w") as f:
        f.write(content)
