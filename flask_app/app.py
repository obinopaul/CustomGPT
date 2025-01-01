import time
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from src.rag.document_loader import DocumentLoader
from src.rag.text_splitter import TextSplitter
from src.rag.utils import get_logger
from typing import List, Optional


app = FastAPI()

class LoadRequest(BaseModel):
    urls: Optional[List[str]] = None
    pdf_files: Optional[List[str]] = None
    docx_files: Optional[List[str]] = None
    github_repos: Optional[List[str]] = None
    splitter_type: Optional[str] = "recursive"
    chunk_size: Optional[int] = 1000
    chunk_overlap: Optional[int] = 200
    model_name: Optional[str] = "gpt-3.5-turbo"

class LoadResponse(BaseModel):
    num_documents: int
    num_chunks: int
    first_chunk: Optional[str]


@app.post("/load_documents", response_model=LoadResponse)
def load_documents(request: LoadRequest):
    try:
        # Load documents
        loader = DocumentLoader(
            urls=request.urls,
            pdf_files=request.pdf_files,
            docx_files=request.docx_files,
            github_repos=request.github_repos
        )
        documents = loader.load_all_documents()
        num_documents = len(documents)

        # Split documents
        splitter = TextSplitter(
            splitter_type=request.splitter_type,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
            model_name=request.model_name if request.splitter_type == "token" else None
        )
        chunks = splitter.split_documents(documents)
        num_chunks = len(chunks)

        # Get first chunk content
        first_chunk = chunks[0].page_content if chunks else None

        return LoadResponse(
            num_documents=num_documents,
            num_chunks=num_chunks,
            first_chunk=first_chunk
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))