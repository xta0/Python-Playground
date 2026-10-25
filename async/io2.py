import asyncio
import time

async def task(name):
    print(f"{name}: start")
    await asyncio.to_thread(time.sleep, 2)  # Non-blocking!
    print(f"{name}: done")

async def main():
    await asyncio.gather(
        task("A"),
        task("B"),
        task("C"),
    )

asyncio.run(main())