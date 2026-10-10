import os
import sys
import asyncio
import time
import httpx


async def send_async_batch(client: httpx.AsyncClient, url: str, batch_size: int):
    tasks = []
    for _ in range(batch_size):
        tasks.append(client.get(url, timeout=5.0))

    try:
        responses = await asyncio.wait_for(
            asyncio.gather(*tasks, return_exceptions=True), timeout=3.0
        )
    except asyncio.TimeoutError:
        return batch_size

    errors = 0
    for resp in responses:
        if isinstance(resp, Exception) or (
            not isinstance(resp, BaseException) and resp.status_code != 200
        ):
            errors += 1
    return errors


async def async_stress_test():
    url = "http://localhost:8000/health"
    print("ASYNCHRONOUS PERFORMANCE STRESS TEST (SLA CONTROL)")

    concurrency_levels = [
        50,
        100,
        250,
        500,
        1000,
        1500,
        2000,
        3000,
        4000,
        5000,
        7500,
        10000,
    ]
    limits = httpx.Limits(max_connections=30000, max_keepalive_connections=10000)

    async with httpx.AsyncClient(limits=limits) as client:
        try:
            await client.get(url, timeout=1.0)
        except Exception:
            print("ERROR: API Server is offline! Run serve-api first.")
            return

        for connections in concurrency_levels:
            start_time = time.time()
            errors = await send_async_batch(client, url, connections)
            duration = time.time() - start_time

            if errors == connections:
                success_rate = 0.0
                print(
                    f"Concurrency: {connections:5} | Batch Time: {duration:6.3f}s | Availability: SLA BREACH (>3.0s)"
                )
            else:
                success_rate = ((connections - errors) / connections) * 100
                print(
                    f"Concurrency: {connections:5} | Batch Time: {duration:6.3f}s | Availability: {success_rate:.1f}%"
                )

            if errors > 0 or success_rate < 100.0:
                print(f"\nCRITICAL: PHYSICAL BREAKING POINT DETECTED!")
                print(
                    f"At concurrency {connections} system breached cumulative SLA capacity threshold."
                )
                print("\n")
                return

    print("SUCCESS: Server capacity successfully verified up to limits.")
    print("\n")


def run_stress_test():
    asyncio.run(async_stress_test())


if __name__ == "__main__":
    run_stress_test()
