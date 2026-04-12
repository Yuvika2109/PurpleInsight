"""FastAPI application entry point."""
from fastapi import FastAPI
from src.api.routes import router

app = FastAPI(title="PurpleInsight — DataTalk")
app.include_router(router)