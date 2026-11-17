from contextlib import asynccontextmanager
import asyncio
import json
import os
from typing import Literal
from uuid import UUID

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, ConfigDict

from .jobs import Conflict, Jobs, TERMINAL
from .settings import ROOT, Settings
from .supervisor import Supervisor, exclusive
from .worker import JobError


class Submission(BaseModel):
    model_config = ConfigDict(extra="forbid")

    request_id: UUID


class Control(Submission):
    action: Literal["cancel", "pause", "export", "stop_export"]


def create_app(settings=None):
    settings = settings or Settings.from_environment()

    @asynccontextmanager
    async def lifespan(app):
        jobs = Jobs(settings.root)
        with exclusive(jobs.state / "app.lock"):
            supervisor = Supervisor(jobs)
            app.state.jobs = jobs
            supervisor.thread.start()
            try:
                yield
            finally:
                supervisor.stop.set()
                supervisor.thread.join()

    app = FastAPI(title="TimbreForge", lifespan=lifespan)
    app.state.settings = settings

    @app.middleware("http")
    async def trusted_local_request(request, call_next):
        if request.url.hostname not in {"127.0.0.1", "localhost", "::1", "testserver"}:
            return JSONResponse({"detail": "服务仅支持本机访问"}, status_code=403)
        origin = request.headers.get("origin")
        if origin and origin != str(request.base_url).rstrip("/"):
            return JSONResponse({"detail": "请求来源不匹配"}, status_code=403)
        return await call_next(request)

    @app.exception_handler(KeyError)
    async def missing(request, exc):
        return JSONResponse({"detail": "记录不存在"}, status_code=404)

    @app.exception_handler(Conflict)
    async def conflict(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=409)

    @app.exception_handler(JobError)
    async def job_error(request, exc):
        return JSONResponse({"detail": str(exc)}, status_code=422)

    @app.get("/api/health")
    def health():
        return {"service": "ready", "name": app.title, "architecture": "dit-1024-16",
                "train_batch_size": 16, "inference_steps": 64}

    @app.post("/api/system/stop")
    def stop(request: Request):
        server = getattr(request.app.state, "server", None)
        if server is None:
            raise HTTPException(409, "请使用项目启动脚本管理该服务")
        store = request.app.state.jobs
        with store.connect(write=True) as db:
            if db.execute("SELECT 1 FROM jobs WHERE state IN ('QUEUED','RUNNING')").fetchone():
                raise Conflict("存在排队或运行中的任务，请先暂停训练或等待任务结束")
            server.should_exit = True
        return {"stopping": True, "pid": os.getpid()}

    @app.get("/api/jobs")
    def jobs(request: Request):
        return request.app.state.jobs.list()

    @app.post("/api/jobs/diagnostic", status_code=202)
    def diagnostic(body: Submission, request: Request):
        return request.app.state.jobs.submit("diagnostic", {}, str(body.request_id))

    @app.get("/api/jobs/{job_id}")
    def job(job_id: str, request: Request):
        return request.app.state.jobs.get(job_id)

    @app.post("/api/jobs/{job_id}/control", status_code=202)
    def control(job_id: str, body: Control, request: Request):
        return request.app.state.jobs.control(job_id, body.action, str(body.request_id))

    @app.get("/api/jobs/{job_id}/events")
    async def events(job_id: str, request: Request):
        store = request.app.state.jobs
        await asyncio.to_thread(store.get, job_id)
        try:
            after = max(0, int(request.headers.get("Last-Event-ID", "0")))
        except ValueError:
            raise HTTPException(400, "Last-Event-ID 无效") from None

        async def stream():
            cursor = after
            while not await request.is_disconnected():
                rows = await asyncio.to_thread(store.events, job_id, cursor)
                for row in rows:
                    cursor = row["id"]
                    data = json.dumps(row, ensure_ascii=False)
                    yield f"id: {cursor}\ndata: {data}\n\n"
                snapshot = await asyncio.to_thread(store.get, job_id)
                if not rows and snapshot["state"] in TERMINAL:
                    yield "event: end\ndata: {}\n\n"
                    return
                if not rows:
                    yield ": heartbeat\n\n"
                await asyncio.sleep(0.5)

        return StreamingResponse(stream(), media_type="text/event-stream",
                                 headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    from .routes import register
    register(app)

    dist = ROOT / "frontend/dist"
    if (dist / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=dist / "assets"), name="assets")

    @app.get("/", include_in_schema=False)
    def index():
        if not (dist / "index.html").is_file():
            raise HTTPException(503, "尚未构建前端，请运行 scripts/setup.ps1")
        return FileResponse(dist / "index.html")

    return app


app = create_app()
