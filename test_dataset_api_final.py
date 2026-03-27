from fastapi.testclient import TestClient
from api.dataset_api import app, TEST_API_KEY
import pytest

client = TestClient(app)

def test_query_dataset_sql_injection():
    # Try an SQL injection on dataset_id
    response = client.get("/datasets/users'; DROP TABLE users;--/metadata", headers={"X-API-Key": TEST_API_KEY})
    print(response.status_code)
    print(response.json())
    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid identifier format: users'; DROP TABLE users;--"
