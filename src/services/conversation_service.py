"""Conversation service module."""

import logging

from src.workflows import AgentWorkFlow

logger = logging.getLogger(__name__)


class ConversationService:
    """Service to execute conversation logic."""

    __agent_workflow: AgentWorkFlow

    def __init__(self, agent_workflow: AgentWorkFlow):
        self.__agent_workflow = agent_workflow

    async def converse(self, message: str):
        """Process the conversation message and return a response."""

        try:
            async for chunk in self.__agent_workflow.stream(prompt=message):
                yield f"data: {chunk}\n\n"

        except Exception as e:
            logger.error(e)
            yield "data: [ERROR]\n\n"
