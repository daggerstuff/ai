with open("tests/utils/test_torch_proxy.py") as f:
    content = f.read()

content = content.replace(
    'mock_torch.__dir__.return_value = ["Tensor", "nn"]',
    'mock_torch.__dir__ = MagicMock(return_value=["Tensor", "nn"])',
)

with open("tests/utils/test_torch_proxy.py", "w") as f:
    f.write(content)
