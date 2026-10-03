import concurrent.futures
from fastapi.testclient import TestClient
from src.api.service import app


def test_contract_valid_input():
    """Тест 1: Валидный контракт входа (Чистые данные)."""
    with TestClient(app) as client:
        payload = {"pixels": [0.0] * 64}
        response = client.post("/predict", json=payload)
        assert response.status_code == 200
        assert "prediction" in response.json()
        assert "is_anomaly" in response.json()


def test_contract_invalid_type():
    """Test 2: Data type violation (string instead of float) -> Code 422."""
    with TestClient(app) as client:
        payload = {"pixels": ["invalid_string"] * 64}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


def test_contract_out_of_range():
    """Test 3: Pixel value outside the 0-16 range -> Code 422."""
    with TestClient(app) as client:
        payload = {"pixels": [25.0] * 64}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


def test_contract_missing_fields():
    """Test 4: Missing required field -> Code 422."""
    with TestClient(app) as client:
        payload = {}
        response = client.post("/predict", json=payload)
        assert response.status_code == 422


def test_load_and_seamless_promotion():
    """Test 5: Seamless failover drill under a load of 100 parallel requests."""
    payload = {"pixels": [0.0] * 64}

    with TestClient(app) as client:

        def send_request():
            return client.post("/predict", json=payload)

        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            futures = [executor.submit(send_request) for _ in range(1000)]

            promo_response = client.post("/promote?version=v2.0.0")
            assert promo_response.status_code == 200

            results = [f.result() for f in futures]

        success_count = sum(1 for r in results if r.status_code == 200)
        assert success_count == 1000
