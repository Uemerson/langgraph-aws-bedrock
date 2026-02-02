"""Main application entry point."""

from src.dependencies import app
from src.routes import conversation_router, knowledgebase_router

app.include_router(conversation_router)
app.include_router(knowledgebase_router)
