"""Upload documents to the knowledge base."""

from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.status import HTTP_400_BAD_REQUEST

from src.dependencies import app
from src.services import KnowledgeBaseService

knowledgebase_router = APIRouter()


@knowledgebase_router.post("/knowledgebase/upload")
async def upload_document(file: UploadFile = File(...)):
    """Upload a document to the knowledge base."""

    if not file.filename.endswith((".pdf", ".docx", ".txt")):
        raise HTTPException(
            status_code=HTTP_400_BAD_REQUEST,
            detail=(
                "Unsupported file type. "
                "Please upload a PDF, DOCX, or TXT file."
            ),
        )

    content = await file.read()
    return KnowledgeBaseService(pinecone=app.state.pinecone).upload_document(
        file.filename, content
    )
