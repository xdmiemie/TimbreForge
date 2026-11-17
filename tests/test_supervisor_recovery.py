import os
import time
from uuid import uuid4

from timbreforge.jobs import Jobs
from timbreforge.supervisor import Supervisor


def test_stale_worker_identity_is_interrupted(tmp_path):
    jobs = Jobs(tmp_path)
    job = jobs.submit("diagnostic", {}, str(uuid4()))
    jobs.claim()
    with jobs.connect(write=True) as db:
        db.execute("UPDATE jobs SET pid=?,process_start=? WHERE id=?", (os.getpid(), 1, job["id"]))
    supervisor = Supervisor(jobs)
    supervisor.thread.start()
    try:
        deadline = time.monotonic() + 5
        while jobs.get(job["id"])["state"] == "RUNNING" and time.monotonic() < deadline:
            time.sleep(0.05)
        assert jobs.get(job["id"])["state"] == "INTERRUPTED"
    finally:
        supervisor.stop.set()
        supervisor.thread.join(5)
