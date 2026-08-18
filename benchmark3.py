import asyncio
import time
import httpx
import numpy as np

async def test_asyncio_gather_vs_list_comprehension():
    async def make_request(i):
        await asyncio.sleep(0.1) # mock network delay
        return 200

    start_time = time.time()
    # Using generator comprehension
    tasks = (make_request(i) for i in range(100))
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    gen_comp_time = end_time - start_time
    print(f"Generator comprehension time: {gen_comp_time:.4f}s")

    start_time = time.time()
    # Using list comprehension
    tasks = [make_request(i) for i in range(100)]
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    list_comp_time = end_time - start_time
    print(f"List comprehension time: {list_comp_time:.4f}s")

asyncio.run(test_asyncio_gather_vs_list_comprehension())
