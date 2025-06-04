# ChatLLM/main.py
from pathlib import Path
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles

from .api import api_router
from .services.chat_service import create_chat

app = FastAPI(title="LLM Chat Backend")
app.include_router(api_router)

# Абсолютный путь до папки static, которая лежит РЯДОМ с ChatLLM
BASE_DIR = Path(__file__).resolve().parent        # .../ChatLLM
STATIC_DIR = BASE_DIR.parent / "static"           # .../static

app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")

@app.on_event("startup")
def _bootstrap():
    create_chat()
