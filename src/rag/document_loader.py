import time
from src.rag.web_scraper import WebScraper
import langchain
from typing import List, Optional, Union, Dict, Optional
import os
import tempfile
import requests
import logging
from io import BytesIO
import shutil
from langchain.docstore.document import Document
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredURLLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    Docx2txtLoader,
)
from langchain_community.document_loaders.parsers import LanguageParser # Import the parser from the community package


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# URL Document Loader
class URLDocumentLoader:
    def __init__(self, url):
        self.urls = url
        self.scraper = WebScraper()
        
    def normalize_url(self, url: str) -> str:
        """
        Normalize the URL by ensuring it has a scheme and removing fragments and query parameters.
        
        :param url: The URL to normalize.
        :return: Normalized URL.
        """
        parsed_url = requests.utils.urlparse(url)
        scheme = parsed_url.scheme or "http"
        netloc = parsed_url.netloc
        path = parsed_url.path
        normalized_url = f"{scheme}://{netloc}{path}"
        return normalized_url

    def is_pdf(self, content_type: str, url: str) -> bool:
        """
        Determine if the content is a PDF based on the Content-Type header or file extension.
        
        :param content_type: The Content-Type header from the HTTP response.
        :param url: The URL being checked.
        :return: True if the content is a PDF, False otherwise.
        """
        if 'application/pdf' in content_type.lower():
            return True
        if url.lower().endswith('.pdf'):
            return True
        return False

    def is_html(self, content_type: str, url: str) -> bool:
        """
        Determine if the content is HTML based on the Content-Type header or file extension.
        
        :param content_type: The Content-Type header from the HTTP response.
        :param url: The URL being checked.
        :return: True if the content is HTML, False otherwise.
        """
        if 'text/html' in content_type.lower():
            return True
        if url.lower().endswith(('.html', '.htm')):
            return True
        return False
    
    def load_from_urls(self) -> List[Dict[str, str]]:
        """
        Load documents from a list of URLs and return a list of dictionaries containing URLs and their content.
        
        :return: List of dictionaries with 'url' and 'content' keys.
        """
        # Fetch the list of URLs using WebScraper
        url_response = self.scraper.api_get_weblinks(self.urls, time_limit=5)
        url_list = url_response.get('urls', [])
        documents = []

        for url in url_list:
            normalized_url = self.normalize_url(url)
            try:
                # Download the content of the URL
                response = requests.get(normalized_url, timeout=10)
                response.raise_for_status()

                # Get Content-Type header
                content_type = response.headers.get('Content-Type', '').lower()

                if self.is_pdf(content_type, normalized_url):
                    # Handle PDF documents
                    with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
                        tmp_file.write(response.content)
                        tmp_file_path = tmp_file.name
                    try:
                        pdf_loader = PyPDFLoader(tmp_file_path)
                        docs = pdf_loader.load()
                        # Extract text from loaded documents
                        for doc in docs:
                            # documents.append({
                            #     'url': normalized_url,
                            #     'content': doc.page_content
                            # })
                            documents.append(Document(
                                page_content=doc.page_content,
                                metadata={'source': normalized_url}
                            ))
                    finally:
                        os.remove(tmp_file_path)  # Clean up the temporary file
                elif self.is_html(content_type, normalized_url):
                    # Handle HTML web pages
                    loader = UnstructuredURLLoader(urls=[normalized_url])
                    docs = loader.load()
                    for doc in docs:
                        # documents.append({
                        #     'url': normalized_url,
                        #     'content': doc.page_content
                        # })
                        documents.append(Document(
                            page_content=doc.page_content,
                            metadata={'source': normalized_url}
                        ))
                else:
                    # Unsupported content type
                    print(f"Unsupported content type for URL {normalized_url}: {content_type}")
            except requests.exceptions.RequestException as e:
                # print(f"Failed to load from URL {normalized_url}: {e}")
                pass
            except Exception as e:
                print(f"An error occurred while processing URL {normalized_url}: {e}")

        return documents


# File Document Loader
class FileDocumentLoader:
    """
    A class to load documents from PDF, DOCX, and TXT files.
    Supports single and multiple file uploads.
    """

    def __init__(self):
        pass

    def load_from_files(self, file_paths: Union[str, List[str]]) -> List[Document]:
        """
        Load documents from a single file or a list of files.

        :param file_paths: A single file path or a list of file paths.
        :return: A list of LangChain Document objects.
        """
        if isinstance(file_paths, str):
            file_paths = [file_paths]

        documents = []

        for file_path in file_paths:
            try:
                # logger.info(f"Processing file: {file_path}")
                file_extension = os.path.splitext(file_path)[1].lower()

                if file_extension == '.pdf':
                    docs = self._load_pdf(file_path)
                elif file_extension in ['.docx', '.doc']:
                    docs = self._load_docx(file_path)
                elif file_extension in ['.txt', '.md', '.log', '.csv', '.tsv', '.rtf', '.yaml', '.json']:
                    docs = self._load_txt(file_path)
                else:
                    logger.warning(f"Unsupported file type for file {file_path}. Skipping.")
                    continue

                documents.extend(docs)
                # logger.info(f"Successfully loaded {len(docs)} documents from {file_path}")

            except Exception as e:
                logger.error(f"Failed to load from file {file_path}: {e}")

        return documents

    def _load_pdf(self, file_path: str) -> List[Document]:
        """
        Load a PDF file and return its pages as Document objects.

        :param file_path: Path to the PDF file.
        :return: A list of Document objects.
        """
        loader = PyPDFLoader(file_path)
        try:
            docs = loader.load()
            return docs
        except Exception as e:
            logger.error(f"Error loading PDF {file_path}: {e}")
            return []

    def _load_docx(self, file_path: str) -> List[Document]:
        """
        Load a DOCX file and return its content as Document objects.

        :param file_path: Path to the DOCX file.
        :return: A list of Document objects.
        """
        try:
            # Try using UnstructuredWordDocumentLoader first
            loader = UnstructuredWordDocumentLoader(file_path)
            docs = loader.load()
            return docs
        except ImportError:
            # Fallback to Docx2txtLoader if Unstructured is not available
            # logger.warning("Unstructured package not found. Falling back to Docx2txtLoader.")
            loader = Docx2txtLoader(file_path)
            docs = loader.load()
            return docs
        except Exception as e:
            logger.error(f"Error loading DOCX {file_path}: {e}")
            return []

    # def _load_txt(self, file_path: str) -> List[Document]:
    #     """
    #     Load a TXT file and return its content as Document objects.

    #     :param file_path: Path to the TXT file.
    #     :return: A list of Document objects.
    #     """
    #     loader = TextLoader(file_path)
        
    #     try:
    #         docs = loader.load()
    #         return docs
    #     except Exception as e:
    #         logger.error(f"Error loading TXT {file_path}: {e}")
    #         return []


    def _load_txt(self, file_path: str) -> List[Document]:
        """
        Load a TXT file and return its content as Document objects.

        :param file_path: Path to the TXT file.
        :return: A list of Document objects.
        """
        try:
            with open(file_path, "r", encoding="utf-8") as file:
                content = file.read()
            # Wrap the content in a Document object
            document = Document(page_content=content, metadata={"source": file_path})
            return [document]
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
            return []
