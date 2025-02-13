# src/rag/text_splitter.py
from typing import List, Optional, Union, Dict
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from src.rag.document_loader import (
    URLDocumentLoader,
    FileDocumentLoader,
)
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TextSplitter:
    """
    A class to split documents into smaller chunks for processing.
    Handles various types of documents loaded via different loaders.
    """
    def __init__(
        self,
        splitter_type: str = "recursive",
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        model_name: Optional[str] = None,
        # ieee_api_key = None,
        # elsevier_api_key = None,
        
        **kwargs
    ):
        """
        Initialize the TextSplitter.
        :param splitter_type: Type of splitter ("recursive" or "token").
        :param chunk_size: Size of each chunk.
        :param chunk_overlap: Overlap between chunks.
        :param model_name: Model name for TokenTextSplitter (e.g., "gpt-3.5-turbo").
        """
        if splitter_type == "token" and not model_name:
            raise ValueError("model_name must be provided for TokenTextSplitter.")
        self.splitter_type = splitter_type
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        # self.ieee_api_key = ieee_api_key
        # self.elsevier_api_key = elsevier_api_key
        # Additional
        self.kwargs = kwargs  # For advanced usage
        if splitter_type == "recursive":
            self.splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        elif splitter_type == "token":
            self.splitter = TokenTextSplitter(
                model_name=model_name,
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap
            )
        else:
            raise ValueError(f"Unsupported splitter_type: {splitter_type}")

    def split_documents(self, documents: List[Document]) -> List[Document]:
        """
        Split a list of documents into smaller chunks.
        :param documents: List of LangChain Document objects.
        :return: List of split Document objects.
        """
        if not documents:
            logger.warning("No documents provided for splitting.")
            return []
        logger.info(f"Splitting {len(documents)} documents using {self.splitter_type} splitter.")
        try:
            split_docs = self.splitter.split_documents(documents)
            logger.info(f"Split into {len(split_docs)} document chunks.")
            return split_docs
        except Exception as e:
            logger.error(f"Error during document splitting: {e}")
            return []

    def split_url_documents(
        self,
        base_url: str,
    ) -> List[Document]:
        """
        Load and split documents from URLs.
        :param base_url: The base URL to start scraping from.
        :param load_max_docs: Maximum number of URL documents to load.
        :return: List of split Document objects.
        """
        logger.info(f"Loading URL documents from base URL: {base_url}")
        loader = URLDocumentLoader(base_url)
        documents = loader.load_from_urls()
        # print(f'successfully loaded {len(documents)} documents')
        return self.split_documents(documents)

    def split_file_documents(
        self,
        file_paths: Union[str, List[str]]
    ) -> List[Document]:
        """
        Load and split documents from file paths.
        :param file_paths: A single file path or a list of file paths.
        :return: List of split Document objects.
        """
        logger.info(f"Loading file documents from paths: {file_paths}")
        loader = FileDocumentLoader()
        documents = loader.load_from_files(file_paths)
        return self.split_documents(documents)
