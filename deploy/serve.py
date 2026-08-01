"""One-origin CPU demo server. Configuration comes from the environment."""
import os
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
os.environ.setdefault("THEARTIST_PUBLIC_HOST", "1")
os.environ.setdefault("THEARTIST_SERVE_FRONTEND", "1")
os.environ.setdefault("THEARTIST_WARMUP", "1")
os.environ.setdefault("THEARTIST_WARMUP_KEYS", "")


def main():
    import uvicorn
    from app.main import app
    from app.services.contracts import capabilities
    from app.services.serving_readiness import require_full_catalog

    started = time.perf_counter()
    models = require_full_catalog()
    print(f"Serving all {len(models)} released model choices (catalog load {time.perf_counter() - started:.3f}s)", flush=True)
    sources = capabilities().data["rag"]["sources"]
    if sources != {"omt": "ready", "berklee": "disabled"}:
        raise SystemExit("Serving requires prepared OMT and disabled Berklee; run backend/scripts/setup_omt.py")
    uvicorn.run(app, host=os.getenv("HOST", "0.0.0.0"), port=int(os.getenv("PORT", "7860")),
                workers=1, limit_concurrency=32, timeout_keep_alive=5, proxy_headers=False)


if __name__ == "__main__":
    main()
