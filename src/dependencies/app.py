"""FastAPI application module."""

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from langchain_aws import ChatBedrockConverse
from pinecone import Pinecone
from pydantic import AnyHttpUrl
from pydantic_settings import BaseSettings

from src.workflows import AgentWorkFlow

logging.basicConfig(
    level=logging.ERROR,
    format="[%(asctime)s] %(levelname)s: %(name)s - %(message)s",
)


class Settings(BaseSettings):
    """Application settings."""

    # API
    BACKEND_CORS_ORIGINS: list[str | AnyHttpUrl] = [
        "*",
    ]

    # AWS Bedrock
    AWS_BEARER_TOKEN_BEDROCK: str
    AWS_ACCESS_KEY_ID: str
    AWS_SECRET_ACCESS_KEY: str

    # PIPECONE
    PINECONE_API_KEY: str

    # LANGCHAIN
    LANGCHAIN_TRACING_V2: bool = True
    LANGCHAIN_API_KEY: str
    LANGCHAIN_PROJECT: str

    class Config:
        """Pydantic configuration for settings."""

        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True


@asynccontextmanager
async def lifespan(app_context: FastAPI) -> AsyncGenerator[None, None]:
    """
    Lifespan context manager for the FastAPI application.
    """

    app_context.state.bedrock_client = ChatBedrockConverse(
        model_id="us.amazon.nova-2-lite-v1:0", region_name="us-east-1"
    )

    app_context.state.pinecone = Pinecone(api_key=settings.PINECONE_API_KEY)
    dense_index_name = "dense-for-hybrid-knowledge-base-index"
    sparse_index_name = "sparse-for-hybrid-knowledge-base-index"

    if not app_context.state.pinecone.has_index(dense_index_name):
        app_context.state.pinecone.create_index_for_model(
            name=dense_index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "llama-text-embed-v2",
                "field_map": {"text": "chunk_text"},
            },
        )

    if not app_context.state.pinecone.has_index(sparse_index_name):
        app_context.state.pinecone.create_index_for_model(
            name=sparse_index_name,
            cloud="aws",
            region="us-east-1",
            embed={
                "model": "pinecone-sparse-english-v0",
                "field_map": {"text": "chunk_text"},
            },
        )

    app_context.state.agent_workflow = AgentWorkFlow(
        client=app_context.state.bedrock_client,
        model_id="us.amazon.nova-2-lite-v1:0",
        pinecone=app_context.state.pinecone,
    )

    yield


settings = Settings()
app = FastAPI(
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[str(origin) for origin in settings.BACKEND_CORS_ORIGINS],
    allow_credentials=True,
    allow_methods=["*"],  # in production, specify allowed methods
    allow_headers=["*"],  # in production, specify allowed headers
)
