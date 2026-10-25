import asyncio, time, threading
from concurrent.futures import ThreadPoolExecutor

def blocking_job(seconds):
    time.sleep(seconds)           # stands in for a GPU run: blocks its thread
    return f"done on {threading.current_thread().name}"

async def ticker(label, t0):
    # a "heartbeat": proves whether the event loop is free
    for _ in range(6):
        print(f"  [{time.perf_counter()-t0:4.2f}s] {label} tick")
        await asyncio.sleep(0.2)

async def bad():
    print("BAD: blocking call directly in a coroutine")
    t0 = time.perf_counter()
    await asyncio.gather(ticker("loop", t0), asyncio.to_thread(lambda: None) if False else _call_blocking())

async def _call_blocking():
    print(f"  {blocking_job(1.0)}")        # no await: freezes the loop for 1s

async def good():
    print("GOOD: run_in_executor with a 1-worker pool")
    t0 = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="worker")
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(pool, blocking_job, 1.0)
    await asyncio.gather(ticker("loop", t0), _wait(fut))

async def _wait(fut):
    print(f"  {await fut}")

async def queueing():
    print("QUEUE: 3 jobs submitted at once to a 1-worker pool")
    t0 = time.perf_counter()
    pool = ThreadPoolExecutor(max_workers=3, thread_name_prefix="worker")
    loop = asyncio.get_running_loop()
    async def one(i):
        await loop.run_in_executor(pool, blocking_job, 0.3)
        print(f"  [{time.perf_counter()-t0:4.2f}s] job {i} finished")
    await asyncio.gather(*(one(i) for i in range(3)))

# asyncio.run(bad()); print()
# asyncio.run(good()); print()
asyncio.run(queueing())