# src/rag/text_splitter.py
from typing import List, Optional, Union, Dict
from langchain.docstore.document import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter, TokenTextSplitter
from src.rag.document_loader import (
    URLDocumentLoader,
    FileDocumentLoader,
    GitHubIssuesDocumentLoader,
    GitHubRepoDocumentLoader,
    ResearchPapersDocumentLoader
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
        model_name: Optional[str] = None
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

    def split_github_issues_documents(
        self,
        repos: Union[str, List[str]],
        access_token: str,
    ) -> List[Document]:
        """
        Load and split documents from GitHub repository issues.
        :param repos: A single repository or a list of repositories in 'owner/repo' format.
        :param access_token: GitHub access token for authentication.
        :param load_max_docs: Maximum number of issues to load.
        :return: List of split Document objects.
        """
        logger.info(f"Loading GitHub issues from repos: {repos}")
        loader = GitHubIssuesDocumentLoader(access_token=access_token)
        documents = loader.load_documents(repos=repos)
        return self.split_documents(documents)

    def split_github_repo_documents(
        self,
        repo_urls: Union[str, List[str]],
        clone_dir: str
    ) -> List[Document]:
        """
        Load and split documents from GitHub repositories.
        :param repo_urls: A single repository URL or a list of repository URLs.
        :param clone_dir: Directory to clone the repositories into.
        :return: List of split Document objects.
        """
        logger.info(f"Loading GitHub repository documents from URLs: {repo_urls}")
        loader = GitHubRepoDocumentLoader(repo_urls=repo_urls, clone_dir=clone_dir)
        documents = loader.load_documents_as_dicts()
        return self.split_documents(documents)

    def split_research_papers_documents(
        self,
        query: str,
        max_results: int = 10,
        sources: Optional[List[str]] = None,
        ieee_api_key: Optional[str] = None,
        elsevier_api_key: Optional[str] = None
    ) -> List[Document]:
        """
        Load and split documents from research paper sources based on a query.
        :param query: The search query string.
        :param max_results: Maximum number of papers to retrieve per source.
        :param sources: List of sources to search in. Options: ['arxiv', 'ieee', 'elsevier']. If None, searches all.
        :param ieee_api_key: IEEE API key for authentication.
        :param elsevier_api_key: Elsevier API key for authentication.
        :return: List of split Document objects.
        """
        logger.info(f"Loading research papers with query: '{query}', max_results: {max_results}, sources: {sources}")
        loader = ResearchPapersDocumentLoader(
            ieee_api_key=ieee_api_key,
            elsevier_api_key=elsevier_api_key
        )
        documents = loader.load_papers_as_dicts(query, max_results, sources)
        return self.split_documents(documents)