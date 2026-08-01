import asyncio
import logging
import os
from pathlib import Path
from collections.abc import Mapping
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.routes.generate import router as generate_router
from app.routes.chords import router as chords_router
from app.routes.models import router as models_router
from app.routes.song import router as song_router
from app.routes.score_import import router as score_import_router
from app.routes.export import router as export_router
from app.routes.plan import router as plan_router
from app.routes.review import router as review_router
from app.routes.reharmonize import router as reharmonize_router
from app.mcp_server import create_mcp_server
from app.services import inference, lockdown
from app.services.contracts import ContractEnvelope, capabilities
from app.services.env_flags import env_bool

# Surface app.* INFO-level logs (model load, rerank breakdown, RAG init,
# etc.) — uvicorn's default config filters them otherwise.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)

WARMUP_VAR = "THEARTIST_WARMUP"
_default_origins = "http://localhost:5173"
FRONTEND_DIST = Path(__file__).resolve().parents[2] / "frontend/dist"


def create_app(env: Mapping[str, str] | None = None) -> FastAPI:
    """The API; on a public host (THEARTIST_PUBLIC_HOST) there is no MCP mount and no MCP session manager."""
    env = os.environ if env is None else env
    mcp_server = None if lockdown.public_host(env) else create_mcp_server()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if mcp_server is None:
            await startup_warmup()
            yield
            return
        async with mcp_server.session_manager.run():
            await startup_warmup()
            yield

    app = FastAPI(title="TheArtist API", version="0.1.0", lifespan=lifespan)

    app.add_middleware(
        CORSMiddleware,
        allow_origins=env.get("CORS_ORIGINS", _default_origins).split(","),
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(generate_router, prefix="/api")
    app.include_router(chords_router, prefix="/api")
    app.include_router(models_router, prefix="/api")
    app.include_router(song_router, prefix="/api")
    app.include_router(score_import_router, prefix="/api")
    app.include_router(export_router, prefix="/api")
    app.include_router(plan_router, prefix="/api")
    app.include_router(review_router, prefix="/api")
    app.include_router(reharmonize_router, prefix="/api")

    @app.get("/api/health")
    def health():
        return {"status": "ok"}

    @app.get("/api/capabilities")
    def api_capabilities() -> ContractEnvelope:
        return capabilities()

    if mcp_server is not None:
        app.mount("/mcp", mcp_server.streamable_http_app())
    else:
        log.info("%s: DAW handoff, the warmup route and MCP are off", lockdown.PUBLIC_HOST_VAR)
    if env_bool("THEARTIST_SERVE_FRONTEND", False, env):
        if not (FRONTEND_DIST / "index.html").is_file():
            raise RuntimeError("Build the frontend before enabling THEARTIST_SERVE_FRONTEND")
        # Only built assets are exposed, never the repository or local source corpora.
        # API routes are registered first so assets cannot shadow them.
        app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True), name="frontend")
    return app


app = create_app()


async def startup_warmup():
    """Preload checkpoints: THEARTIST_WARMUP=1 loads every one, THEARTIST_WARMUP_KEYS only the listed ones."""
    keys, unknown = inference.warmup_keys()
    # A key list turns the preload on by default; an explicit THEARTIST_WARMUP=0 still wins
    if not env_bool(WARMUP_VAR, keys is not None):
        return
    if unknown:
        log.warning("%s names no checkpoint: %s (known keys: %s)", inference.WARMUP_KEYS_VAR, ", ".join(unknown),
                    ", ".join(inference.CHECKPOINTS))
    if inference.TORCH_AVAILABLE:
        loop = asyncio.get_running_loop()
        loaded = await loop.run_in_executor(None, inference.ModelRegistry.get().warmup, keys)
        log.info("warmup loaded %d of %d checkpoint(s)", sum(loaded.values()), len(loaded))
