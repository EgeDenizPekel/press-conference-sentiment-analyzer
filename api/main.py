"""FastAPI application entry point."""
from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api import data as data_module
from api.routers import analysis, series, speakers


@asynccontextmanager
async def lifespan(app: FastAPI):
    data_module.app_data = data_module.load()
    yield


app = FastAPI(title="Press Conference Sentiment Analyzer", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

app.include_router(analysis.router, prefix="/analysis", tags=["analysis"])
app.include_router(series.router, prefix="/series", tags=["series"])
app.include_router(speakers.router, prefix="/speakers", tags=["speakers"])


@app.get("/")
def root():
    return {"status": "ok", "message": "Press Conference Sentiment Analyzer API"}
