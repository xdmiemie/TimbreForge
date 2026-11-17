import time

from fastapi.testclient import TestClient

from timbreforge.api import create_app
from timbreforge.settings import Settings


def test_worker_resumes_event_observation(tmp_path):
    import uuid

    app = create_app(Settings(tmp_path))
    with TestClient(app) as client:
        body = {"request_id": str(uuid.uuid4())}
        first = client.post("/api/jobs/diagnostic", json=body).json()
        second = client.post("/api/jobs/diagnostic", json=body).json()
        assert first["id"] == second["id"]
        path = "/api/jobs/" + first["id"]
        deadline = time.monotonic() + 15
        while time.monotonic() < deadline:
            if client.get(path).json()["state"] == "COMPLETED":
                break
            time.sleep(0.1)
        assert client.get(path).json()["state"] == "COMPLETED"
        stream = client.get(path + "/events")
        ids = [int(line[4:]) for line in stream.text.splitlines() if line.startswith("id: ")]
        assert len(ids) == 7
        resumed = client.get(path + "/events", headers={"Last-Event-ID": str(ids[-2])})
        assert [line for line in resumed.text.splitlines() if line.startswith("id: ")] == [f"id: {ids[-1]}"]
