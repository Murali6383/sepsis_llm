"""
SepsisGuardian AI — FastAPI Application Entry Point
====================================================
Start with: uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
(from the backend/ directory)
"""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.api.routes import ai, auth, predictions
from app.core.config import get_settings

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
)
logger = logging.getLogger("main")

# ── Lifespan ───────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("SepsisGuardian AI backend starting up …")
    from app.services.ml_service import get_ml_service
    ml = get_ml_service()
    available = ml.models_available()
    logger.info(f"ML model availability at startup: {available}")
    missing = [k for k, v in available.items() if not v]
    if missing:
        logger.warning(
            f"Missing models: {missing}. "
            "Run: cd backend && python -m app.models.train_models"
        )
    yield
    logger.info("SepsisGuardian AI backend shutting down …")

# ── App ───────────────────────────────────────────────────────────────────────
settings = get_settings()

app = FastAPI(
    title="SepsisGuardian AI",
    description=(
        "Explainable AI Framework for Sepsis Risk Prediction, "
        "Multi-Organ Dysfunction Assessment, Antibiotic Susceptibility Analysis, "
        "and Groq LLM-Based Clinical Research Explanations.\n\n"
        "Data source: ARMD-ECUH retrospective research dataset for antimicrobial susceptibility "
        "and research benchmark prototypes for sepsis/organ dysfunction."
    ),
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.allowed_origins_list,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Global Error Handler ──────────────────────────────────────────────────────
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception on {request.url}: {exc}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={"detail": "An internal server error occurred. Please try again."},
    )

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(auth.router)
app.include_router(predictions.router)
app.include_router(ai.router)

# ── Health ────────────────────────────────────────────────────────────────────
@app.get("/health", tags=["System"])
def health():
    return {"status": "ok", "service": "SepsisGuardian AI"}
