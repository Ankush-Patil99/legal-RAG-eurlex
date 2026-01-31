from fastapi import FastAPI
from api.routes import router
from api.logging import setup_logging

# Initialize logging once at app startup
setup_logging()

app = FastAPI(title="Legal RAG System")

app.include_router(router)
