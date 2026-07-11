import asyncio
import time
import httpx
import numpy as np
from infrastructure.qa.enterprise_validator import EnterpriseValidator

async def test_gather_generator():
    validator = EnterpriseValidator()

    # Mocking client requests
    class MockClient:
        async def get(self, *args, **kwargs):
            await asyncio.sleep(0.01)
            class MockResponse:
                status_code = 200
            return MockResponse()

        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass

    import httpx
    httpx.AsyncClient = MockClient

    start_time = time.time()
    await validator._run_security_validation()
    await validator._run_performance_validation()
    end_time = time.time()
    print(f"Total validation time: {end_time - start_time:.4f}s")


asyncio.run(test_gather_generator())
