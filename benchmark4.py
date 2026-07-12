import asyncio
import time
import httpx
import numpy as np

async def test_asyncio_gather_vs_list_comprehension():
    async def make_request(i):
        await asyncio.sleep(0.1) # mock network delay
        return 200

    start_time = time.time()
    # Using generator expression directly in gather
    results = await asyncio.gather(*(make_request(i) for i in range(100)))
    end_time = time.time()
    gen_comp_time = end_time - start_time
    print(f"Generator comprehension time inline: {gen_comp_time:.4f}s")

    start_time = time.time()
    # Using list comprehension directly in gather
    results = await asyncio.gather(*[make_request(i) for i in range(100)])
    end_time = time.time()
    list_comp_time = end_time - start_time
    print(f"List comprehension time inline: {list_comp_time:.4f}s")

    start_time = time.time()
    # Create tasks explicitly first
    tasks = [asyncio.create_task(make_request(i)) for i in range(100)]
    results = await asyncio.gather(*tasks)
    end_time = time.time()
    create_task_time = end_time - start_time
    print(f"create_task time inline: {create_task_time:.4f}s")


asyncio.run(test_asyncio_gather_vs_list_comprehension())
