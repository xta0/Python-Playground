from concurrent.futures import ThreadPoolExecutor

def cpu_task():
    total = 0
    for i in range(30_000_000):
        total += i * i
    return total

with ThreadPoolExecutor(max_workers=2) as executor:
    a = executor.submit(cpu_task)
    b = executor.submit(cpu_task)

    a.result()
    b.result()