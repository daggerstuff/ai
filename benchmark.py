import asyncio
import time
import httpx
from infrastructure.qa.enterprise_validator import SecurityValidator, PerformanceValidator

async def main():
    validator = SecurityValidator()

    start_time = time.time()
    await validator.run_rate_limiting_test()
    end_time = time.time()
    print(f"Rate Limiting Test: {end_time - start_time:.4f}s")

    validator2 = PerformanceValidator()
    start_time = time.time()
    await validator2.run_response_time_test()
    end_time = time.time()
    print(f"Response Time Test: {end_time - start_time:.4f}s")

if __name__ == "__main__":
    asyncio.run(main())
