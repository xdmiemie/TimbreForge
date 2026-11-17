from contextlib import contextmanager
import json
from pathlib import Path
import sqlite3
import time
import uuid


ACTIVE = ("QUEUED", "RUNNING")
TERMINAL = ("COMPLETED", "FAILED", "INTERRUPTED", "CANCELLED", "PAUSED", "EARLY_COMPLETED")


class Conflict(ValueError):
    pass


class Jobs:
    def __init__(self, root: Path):
        self.root = root
        self.state = root / "state"
        self.state.mkdir(parents=True, exist_ok=True)
        self.db = self.state / "timbreforge.sqlite"
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY, request TEXT UNIQUE NOT NULL,
                    kind TEXT NOT NULL, state TEXT NOT NULL, payload TEXT NOT NULL,
                    result TEXT, error TEXT, created REAL NOT NULL, updated REAL NOT NULL,
                    pid INTEGER, process_start REAL, instance TEXT
                );
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                    kind TEXT NOT NULL, payload TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE INDEX IF NOT EXISTS job_events ON events(job_id, id);
                CREATE TABLE IF NOT EXISTS controls (
                    request TEXT PRIMARY KEY, job_id TEXT NOT NULL REFERENCES jobs(id),
                    action TEXT NOT NULL, created REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS control_receipts (
                    request TEXT PRIMARY KEY REFERENCES controls(request)
                );
                CREATE TABLE IF NOT EXISTS search_receipts (
                    id TEXT PRIMARY KEY, candidates TEXT NOT NULL, expires REAL NOT NULL
                );
                CREATE TABLE IF NOT EXISTS current_cover (
                    id INTEGER PRIMARY KEY CHECK(id=1), job_id TEXT NOT NULL REFERENCES jobs(id)
                );
                CREATE TABLE IF NOT EXISTS cover_cleanup (
                    job_id TEXT PRIMARY KEY REFERENCES jobs(id),
                    attempts INTEGER NOT NULL DEFAULT 0, retry_at REAL NOT NULL DEFAULT 0
                );
                CREATE TABLE IF NOT EXISTS saved_covers (
                    id TEXT PRIMARY KEY, metadata TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS model_selection (
                    id INTEGER PRIMARY KEY CHECK(id=1), job_id TEXT REFERENCES jobs(id)
                );
                PRAGMA user_version=2;
            """)

    @contextmanager
    def connect(self, write=False):
        db = sqlite3.connect(self.db, timeout=15)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            if write:
                db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    @staticmethod
    def decode(row):
        if row is None:
            raise KeyError("任务不存在")
        item = dict(row)
        for field in ("payload", "result", "latest_progress"):
            if item.get(field) is not None:
                item[field] = json.loads(item[field])
        return item

    @staticmethod
    def emit(db, job_id, kind, payload):
        db.execute("INSERT INTO events(job_id,kind,payload,created) VALUES(?,?,?,?)",
                   (job_id, kind, json.dumps(payload, ensure_ascii=False, allow_nan=False), time.time()))

    def submit(self, kind, payload, request, *, on_accept=None):
        snapshot = json.dumps(payload, ensure_ascii=False, sort_keys=True, allow_nan=False)
        with self.connect(write=True) as db:
            existing = db.execute("SELECT * FROM jobs WHERE request=?", (request,)).fetchone()
            if existing:
                if existing["kind"] != kind or existing["payload"] != snapshot:
                    raise Conflict("该请求 ID 已绑定其他配置")
                return self.decode(existing)
            job_id, now = uuid.uuid4().hex, time.time()
            db.execute("INSERT INTO jobs(id,request,kind,state,payload,created,updated) VALUES(?,?,?,?,?,?,?)",
                       (job_id, request, kind, "QUEUED", snapshot, now, now))
            if on_accept:
                on_accept(db, job_id)
            self.emit(db, job_id, "state", {"state": "QUEUED"})
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def get(self, job_id):
        with self.connect() as db:
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())

    def list(self):
        with self.connect() as db:
            return [self.decode(row) for row in db.execute("""
                SELECT j.*, (SELECT e.payload FROM events e WHERE e.job_id=j.id AND e.kind='progress'
                             ORDER BY e.id DESC LIMIT 1) AS latest_progress
                FROM jobs j ORDER BY j.created DESC LIMIT 100
            """)]

    def running(self):
        with self.connect() as db:
            return [self.decode(row) for row in db.execute("SELECT * FROM jobs WHERE state='RUNNING'")]

    def claim(self):
        with self.connect(write=True) as db:
            row = db.execute("SELECT * FROM jobs WHERE state='QUEUED' ORDER BY created LIMIT 1").fetchone()
            if row is None:
                return None
            instance = uuid.uuid4().hex
            db.execute("UPDATE jobs SET state='RUNNING',instance=?,updated=? WHERE id=?",
                       (instance, time.time(), row["id"]))
            self.emit(db, row["id"], "state", {"state": "RUNNING"})
            return {**self.decode(row), "instance": instance, "state": "RUNNING"}

    def status(self, job_id, state, *, result=None, error=None, instance=None):
        with self.connect(write=True) as db:
            changed = db.execute(
                """UPDATE jobs SET state=?,result=?,error=?,updated=?
                   WHERE id=? AND state='RUNNING' AND (instance=? OR ? IS NULL)""",
                (state, json.dumps(result, ensure_ascii=False, allow_nan=False) if result is not None else None,
                 error, time.time(), job_id, instance, instance),
            )
            if changed.rowcount:
                self.emit(db, job_id, "state", {"state": state, "error": error})

    def progress(self, job_id, payload):
        with self.connect(write=True) as db:
            self.emit(db, job_id, "progress", payload)
            db.execute("UPDATE jobs SET updated=? WHERE id=?", (time.time(), job_id))

    def events(self, job_id, after):
        with self.connect() as db:
            return [{**dict(row), "payload": json.loads(row["payload"])} for row in db.execute(
                "SELECT * FROM events WHERE job_id=? AND id>? ORDER BY id LIMIT 100", (job_id, after))]

    def control(self, job_id, action, request):
        with self.connect(write=True) as db:
            current = self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
            existing = db.execute("SELECT * FROM controls WHERE request=?", (request,)).fetchone()
            if existing:
                if existing["job_id"] != job_id or existing["action"] != action:
                    raise Conflict("该控制请求 ID 已使用")
                return dict(existing)
            if current["state"] not in ACTIVE:
                raise Conflict("当前任务已结束")
            if action != "cancel" and (current["kind"] != "train" or current["state"] != "RUNNING"):
                raise Conflict("此控制操作需要正在运行的训练任务")
            if action != "cancel" and db.execute(
                """SELECT 1 FROM controls c LEFT JOIN control_receipts r ON c.request=r.request
                   WHERE c.job_id=? AND r.request IS NULL AND c.action IN ('pause','stop_export')""",
                (job_id,),
            ).fetchone():
                raise Conflict("训练正在等待暂停或中止")
            now = time.time()
            db.execute("INSERT INTO controls VALUES(?,?,?,?)", (request, job_id, action, now))
            if current["state"] == "QUEUED":
                db.execute("UPDATE jobs SET state='CANCELLED',updated=? WHERE id=?", (now, job_id))
                self.emit(db, job_id, "state", {"state": "CANCELLED"})
            else:
                self.emit(db, job_id, "control", {"action": action, "request": request})
            return {"request": request, "job_id": job_id, "action": action, "created": now}

    def cancelled(self, job_id):
        with self.connect() as db:
            return db.execute("SELECT 1 FROM controls WHERE job_id=? AND action='cancel'", (job_id,)).fetchone() is not None

    def pending_control(self, job_id):
        with self.connect() as db:
            return db.execute(
                """SELECT c.* FROM controls c LEFT JOIN control_receipts r ON c.request=r.request
                   WHERE c.job_id=? AND r.request IS NULL AND c.action!='cancel' ORDER BY c.created LIMIT 1""",
                (job_id,),
            ).fetchone()

    def complete_control(self, job_id, request):
        with self.connect(write=True) as db:
            db.execute("INSERT OR IGNORE INTO control_receipts VALUES(?)", (request,))
            self.emit(db, job_id, "control_applied", {"request": request})

    def retry_cover(self, job_id, request):
        with self.connect(write=True) as db:
            row = self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
            existing = db.execute("SELECT * FROM controls WHERE request=?", (request,)).fetchone()
            if existing:
                if existing["job_id"] != job_id or existing["action"] != "retry":
                    raise Conflict("该请求 ID 已使用")
                return row
            pointer = db.execute("SELECT job_id FROM current_cover WHERE id=1").fetchone()
            if (row["kind"] != "cover" or not pointer or pointer["job_id"] != job_id
                    or row["state"] not in {"FAILED", "INTERRUPTED"}):
                raise Conflict("当前翻唱任务无法重试")
            now = time.time()
            db.execute("INSERT INTO controls VALUES(?,?,?,?)", (request, job_id, "retry", now))
            db.execute("UPDATE jobs SET state='QUEUED',error=NULL,pid=NULL,process_start=NULL,instance=NULL,updated=? WHERE id=?",
                       (now, job_id))
            self.emit(db, job_id, "state", {"state": "QUEUED", "retry": True})
            return self.decode(db.execute("SELECT * FROM jobs WHERE id=?", (job_id,)).fetchone())
