from contextlib import contextmanager
import os
import subprocess
import sys
from threading import Event, Thread

@contextmanager
def exclusive(path):
    with path.open("a+b") as lock:
        lock.seek(0)
        if os.name == "nt":
            import msvcrt

            msvcrt.locking(lock.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl

            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        yield


def live_identity(job):
    import psutil

    if job["pid"] is None or job["process_start"] is None:
        return False
    try:
        process = psutil.Process(job["pid"])
        return process.create_time() == job["process_start"] and process.status() != psutil.STATUS_ZOMBIE
    except psutil.NoSuchProcess:
        return False
    except psutil.AccessDenied:
        return True


class Supervisor:
    def __init__(self, jobs):
        self.jobs = jobs
        self.stop = Event()
        self.thread = Thread(target=self.run, name="job-supervisor", daemon=True)

    def run(self):
        while not self.stop.is_set():
            try:
                with exclusive(self.jobs.state / "worker.lock"):
                    from .covers import cleanup
                    from .library import cleanup_deleted

                    cleanup(self.jobs)
                    cleanup_deleted(self.jobs)
                    running = self.jobs.running()
                    for job in running:
                        if not live_identity(job):
                            self.jobs.status(job["id"], "INTERRUPTED", error="工作进程已退出")
                    busy = any(live_identity(job) for job in running)
                    job = None if busy else self.jobs.claim()
                    if job:
                        self.launch(job)
            except OSError:
                pass
            self.stop.wait(0.2)

    def launch(self, job):
        import psutil

        job_id = job["id"]
        command = [sys.executable, "-B", "-m", "timbreforge.worker",
                   str(self.jobs.root), job_id, job["instance"]]
        # Do not copy credential values into non-network workers.
        env = {key: value for key, value in os.environ.items()
               if key not in ("ASTRBOT_CHAT_NETEASE_COOKIE", "ALYSIA_NETEASE_COOKIE")}
        if job["kind"] == "cover" and job["payload"].get("song"):
            for key in ("ASTRBOT_CHAT_NETEASE_COOKIE", "ALYSIA_NETEASE_COOKIE"):
                if key in os.environ:
                    env[key] = os.environ[key]
        try:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                       stderr=subprocess.DEVNULL, env=env)
        except OSError:
            self.jobs.status(job_id, "FAILED", error="无法启动工作进程")
            return
        try:
            identity = psutil.Process(process.pid).create_time()
            with self.jobs.connect(write=True) as db:
                db.execute("UPDATE jobs SET pid=?,process_start=? WHERE id=?", (process.pid, identity, job_id))
        except psutil.NoSuchProcess:
            self.jobs.status(job_id, "INTERRUPTED", error="工作进程启动后退出")
        self.stop.wait(0.05)
        # The worker retains its own lease after the supervisor drops the startup lease.
        Thread(target=process.wait, daemon=True).start()
