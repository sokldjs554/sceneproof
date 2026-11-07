"""FastAPI application wiring: store, VLM worker, gRPC ingest, static console."""

from __future__ import annotations

import asyncio
import logging
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from .api import build_router
from .config import ServerSettings
from .db import Store
from .ingest import CommandHub, IngestService, serve_grpc
from .replay import import_bundle
from .vlm.factory import build_provider
from .vlm.worker import VLMWorker

log = logging.getLogger(__name__)


def create_app(settings: ServerSettings | None = None) -> FastAPI:
    s = settings or ServerSettings()
    Path(s.data_dir).mkdir(parents=True, exist_ok=True)
    store = Store(s.resolved_db())
    provider = build_provider(s)
    worker = VLMWorker(store, provider, s)
    hub = CommandHub(store)
    ingest = IngestService(store, s.resolved_evidence(), worker, hub)
    state = SimpleNamespace(settings=s, store=store, worker=worker, hub=hub, ingest=ingest, grpc=None)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if s.replay_bundle:
            stats = import_bundle(store, s.replay_bundle, s.resolved_evidence(), worker)
            log.info("replay bundle imported: %s", stats)
        await worker.start()
        if s.mode != "replay":
            state.grpc = await serve_grpc(ingest, s.grpc_listen)
        log.info("zonewatch server up: mode=%s vlm=%s http=%s:%s", s.mode, provider.name if provider else "none",
                 s.http_host, s.http_port)
        try:
            yield
        finally:
            await worker.stop()
            if state.grpc is not None:
                await state.grpc.stop(grace=2)
            store.close()

    app = FastAPI(title="ZoneWatch review server", lifespan=lifespan)
    app.state.zw = state
    app.include_router(build_router(state))
    console = Path(s.console_dir)
    if console.exists():
        @app.get("/")
        def index():
            return FileResponse(str(console / "index.html"))

        app.mount("/", StaticFiles(directory=str(console), html=True), name="console")
    return app


def run(settings: ServerSettings | None = None) -> None:
    import uvicorn

    s = settings or ServerSettings()
    app = create_app(s)
    uvicorn.run(app, host=s.http_host, port=s.http_port, log_level="info")


async def _unused():  # keep asyncio import meaningful for type checkers
    await asyncio.sleep(0)
