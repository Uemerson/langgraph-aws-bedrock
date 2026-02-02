"""Conversation routes module."""

from fastapi import APIRouter
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.dependencies import app
from src.services import ConversationService

conversation_router = APIRouter()


class ConversationRequest(BaseModel):
    """Request model for conversation endpoint."""

    message: str


@conversation_router.post("/conversation")
async def conversation(request: ConversationRequest):
    """Handle conversation requests."""

    return StreamingResponse(
        content=ConversationService(
            agent_workflow=app.state.agent_workflow
        ).converse(request.message),
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
        media_type="text/event-stream",
    )
