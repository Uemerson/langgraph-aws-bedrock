"""
Knowledge Base Service Module
"""

import uuid
from io import BytesIO

from docx import Document as DocxDocument
from fastapi.responses import JSONResponse
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import Pinecone
from pypdf import PdfReader


class KnowledgeBaseService:
    """Service to handle knowledge base operations."""

    def __init__(self, pinecone: Pinecone):
        self.__pinecone = pinecone

    def upload_document(self, filename: str, content: bytes):
        """Upload a document to the knowledge base.

        Args:
            filename (str): The name of the file.
            content (bytes): The content of the file.
        """

        try:
            text = self._extract_text(filename, content)

            if not text.strip():
                raise ValueError("No readable text found in document")

            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50,
            )

            texts = text_splitter.split_text(text)
            records = [
                {
                    "_id": str(uuid.uuid4()),
                    "chunk_text": chunk,
                    "filename": filename,
                }
                for chunk in texts
            ]
            dense_index = self.__pinecone.Index(
                "dense-for-hybrid-knowledge-base-index"
            )
            sparse_index = self.__pinecone.Index(
                "sparse-for-hybrid-knowledge-base-index"
            )
            dense_index.upsert_records("rag-namespace", records)
            sparse_index.upsert_records("rag-namespace", records)

            return JSONResponse(content={}, status_code=200)
        except Exception as e:
            print(e)
            return JSONResponse(
                content={
                    "detail": "Internal Server Error",
                },
                status_code=500,
            )

    def _extract_text(self, filename: str, content: bytes) -> str:
        ext = filename.lower().split(".")[-1]

        if ext == "pdf":
            return self._read_pdf(content)

        if ext == "docx":
            return self._read_docx(content)

        if ext == "txt":
            return self._read_txt(content)

        raise ValueError("Unsupported file type")

    def _read_pdf(self, content: bytes) -> str:
        reader = PdfReader(BytesIO(content))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    def _read_docx(self, content: bytes) -> str:
        doc = DocxDocument(BytesIO(content))
        return "\n".join(
            para.text for para in doc.paragraphs if para.text.strip()
        )

    def _read_txt(self, content: bytes) -> str:
        return content.decode("utf-8", errors="ignore")
