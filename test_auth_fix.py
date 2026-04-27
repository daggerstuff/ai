import sys
import asyncio
from api.dataset_api import get_current_active_user_or_api_key, list_datasets, get_dataset_metadata, query_dataset, app
from fastapi.testclient import TestClient

client = TestClient(app)

print("Imports passed")
