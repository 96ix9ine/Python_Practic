import asyncio
import concurrent.futures
import httpx

BASE_URL = "http://localhost:8000"
TIMEOUT_CFG = httpx.Timeout(30.0, connect=5.0)


def test_contract_valid_input():
    """Test 1: Valid API contract with clean data."""
    payload = {"pixels": [0.0] * 64}
    response = httpx.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT_CFG)
    assert response.status_code == 200


def test_contract_invalid_type():
    """Test 2: Invalid data type contract validation."""
    payload = {"pixels": ["invalid_string"] * 64}
    response = httpx.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT_CFG)
    assert response.status_code == 422


def test_contract_out_of_range():
    """Test 3: Value out of range contract validation."""
    payload = {"pixels": [25.0] * 64}
    response = httpx.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT_CFG)
    assert response.status_code == 422


def test_contract_missing_fields():
    """Test 4: Missing fields contract validation."""
    payload = {}
    response = httpx.post(f"{BASE_URL}/predict", json=payload, timeout=TIMEOUT_CFG)
    assert response.status_code == 422


def test_load_and_intercept_anomalies():
    """Test 5: Inject OOD anomalies and handle seamless promotion load."""
    payload_anomaly = {"pixels": [16.0] * 64}
    for _ in range(15):
        httpx.post(f"{BASE_URL}/predict", json=payload_anomaly, timeout=TIMEOUT_CFG)

    payload_clean = {"pixels": [0.0] * 64}

    async def send_async_requests():
        async with httpx.AsyncClient(timeout=TIMEOUT_CFG) as client:
            tasks = [
                client.post(f"{BASE_URL}/predict", json=payload_clean)
                for _ in range(50)
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

    with concurrent.futures.ThreadPoolExecutor() as executor:
        promo_resp = httpx.post(
            f"{BASE_URL}/promote?version=v2.0.0", timeout=TIMEOUT_CFG
        )
        assert promo_resp.status_code == 200

        future_requests = executor.submit(lambda: asyncio.run(send_async_requests()))
        results = future_requests.result()

    success_count = sum(
        1 for r in results if not isinstance(r, Exception) and r.status_code == 200
    )
    assert success_count >= 10


def test_z_run_performance_stress():
    """Test 6: Run performance stress to trigger metrics latency peak."""
    from src.tests.test_stress import run_stress_test

    run_stress_test()
