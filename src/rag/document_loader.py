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
import arxiv
from getpass import getpass
from langchain.docstore.document import Document
from langchain_community.document_loaders.generic import GenericLoader
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredURLLoader,
    GitLoader,
    DirectoryLoader,
    TextLoader,
    UnstructuredWordDocumentLoader,
    Docx2txtLoader,
    GitHubIssuesLoader,
    ArxivLoader
)
from langchain_community.document_loaders.parsers import LanguageParser # Import the parser from the community package
from git import Repo, GitCommandError


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
                elif file_extension in ['.txt', '.md']:
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

    def _load_txt(self, file_path: str) -> List[Document]:
        """
        Load a TXT file and return its content as Document objects.

        :param file_path: Path to the TXT file.
        :return: A list of Document objects.
        """
        loader = TextLoader(file_path)
        try:
            docs = loader.load()
            return docs
        except Exception as e:
            logger.error(f"Error loading TXT {file_path}: {e}")
            return []


# GitHub Issues Document Loader
class GitHubIssuesDocumentLoader:
    """
    A class to load documents from GitHub repository issues.
    Supports single and multiple repositories, with options to include/exclude pull requests and specify issue states.
    """

    def __init__(
        self,
        access_token: Optional[str] = None, # GitHub Personal Access Token
        include_prs: bool = False,
        state: str = "all",  # Can be 'open', 'closed', or 'all'
    ):
        """
        Initialize the GitHubIssuesDocumentLoader.
        
        :param access_token: GitHub Personal Access Token. If not provided, prompts the user.
        :param include_prs: Whether to include pull requests in the loaded issues.
        :param state: The state of issues to load ('open', 'closed', 'all').
        """
        self.access_token = access_token or self._prompt_for_token()
        self.include_prs = include_prs
        self.state = state.lower()
        if self.state not in ["open", "closed", "all"]:
            raise ValueError("State must be one of 'open', 'closed', or 'all'.")
    
    def _prompt_for_token(self) -> str:
        """
        Prompt the user to enter their GitHub Personal Access Token securely.
        
        :return: The entered GitHub token.
        """
        return getpass("Enter your GitHub Personal Access Token: ")
    
    def load_issues_from_repo(
        self,
        repo: str
    ) -> List[Document]:
        """
        Load issues from a single GitHub repository.
        
        :param repo: The repository in the format 'owner/repo', e.g., 'huggingface/peft'.
        :return: A list of LangChain Document objects representing the issues.
        """
        logger.info(f"Loading issues from repository: {repo}")
        loader = GitHubIssuesLoader(
            repo=repo,
            access_token=self.access_token,
            include_prs=self.include_prs,
            state=self.state
        )
        try:
            docs = loader.load()
            logger.info(f"Loaded {len(docs)} issues from {repo}")
            return docs
        except Exception as e:
            logger.error(f"Failed to load issues from {repo}: {e}")
            return []
    
    def load_issues_from_repos(
        self,
        repos: Union[str, List[str]]
    ) -> List[Document]:
        """
        Load issues from one or multiple GitHub repositories.
        
        :param repos: A single repository string or a list of repository strings in the format 'owner/repo'.
        :return: A combined list of LangChain Document objects from all repositories.
        """
        if isinstance(repos, str):
            repos = [repos]
        
        all_documents = []
        for repo in repos:
            docs = self.load_issues_from_repo(repo)
            all_documents.extend(docs)
        
        logger.info(f"Total issues loaded from all repositories: {len(all_documents)}")
        return all_documents
    
    def load_documents(
        self,
        repos: Union[str, List[str]]
    ) -> List[Dict[str, Union[str, Dict[str, str]]]]:
        """
        Load documents from GitHub issues and return them as a list of dictionaries containing URL and content.
        
        :param repos: A single repository string or a list of repository strings in the format 'owner/repo'.
        :return: A list of dictionaries with 'url' and 'content' keys.
        """
        docs = self.load_issues_from_repos(repos)
        document_list = []
        
        for doc in docs:
            # Assuming each Document has metadata including 'url'
            url = doc.metadata.get('url', 'No URL')
            title = doc.metadata.get('title', 'No Title')
            body = doc.page_content or ''
            content = f"Title: {title}\n\n{body}"
            
            # document_list.append({
            #     'url': url,
            #     'content': content
            # })
            
            document_list.append(Document(
                page_content=content,
                metadata={'source': url}
            ))
        
        logger.info(f"Total documents prepared: {len(document_list)}")
        return document_list



class GitHubRepoDocumentLoader:
    """
    A class to load documents from one or multiple GitHub repositories.
    Supports multiple file types with automatic encoding detection.
    """

    def __init__(
        self,
        repo_urls: Union[str, List[str]],
        include_prs: bool = False,
        clone_dir: Optional[str] = None,
        glob_pattern: str = "**/*",
        parser: Optional[LanguageParser] = None,
        file_extensions: Optional[List[str]] = None
    ):
        """
        Initialize the GitHubRepoDocumentLoader.

        :param repo_urls: A single repository URL or a list of repository URLs.
        :param include_prs: Whether to include pull requests.
        :param clone_dir: Directory to clone repositories into. If None, uses temporary directories.
        :param glob_pattern: Glob pattern to match files within repositories.
        :param parser: Parser to use with GenericLoader.
        :param file_extensions: List of file extensions to process (e.g., ['.txt', '.md', '.py'])
        """
        if isinstance(repo_urls, str):
            self.repo_urls = [repo_urls]
        else:
            self.repo_urls = repo_urls

        self.include_prs = include_prs
        self.clone_dir = clone_dir
        self.glob_pattern = glob_pattern
        self.parser = parser or LanguageParser()
        self.file_extensions = file_extensions or ['.txt', '.md', '.py', '.js', '.java', '.c', '.cpp', '.h', '.cs', '.rb']
        self.cloned_repos = []

        # Validate repository URLs
        for url in self.repo_urls:
            if not url.startswith(('http://', 'https://')):
                raise ValueError(f"Invalid repository URL format: {url}")

    def _read_file_with_encoding(self, file_path: str) -> Optional[str]:
        """
        Read a file with appropriate encoding detection.
        
        :param file_path: Path to the file to read
        :return: File content as string or None if reading fails
        """
        try:
            # First try UTF-8
            with open(file_path, 'r', encoding='utf-8') as f:
                return f.read()
        except UnicodeDecodeError:
            try:
                # If UTF-8 fails, detect encoding
                with open(file_path, 'rb') as f:
                    raw_data = f.read()
                
                # Detect the file encoding
                result = chardet.detect(raw_data)
                encoding = result['encoding']
                
                if encoding is None:
                    logger.warning(f"Could not detect encoding for {file_path}")
                    return None
                
                return raw_data.decode(encoding)
            except Exception as e:
                logger.warning(f"Failed to read {file_path}: {e}")
                return None

    def should_process_file(self, file_path: str) -> bool:
        """
        Determine if a file should be processed based on its extension.
        
        :param file_path: Path to the file to check
        :return: Boolean indicating whether to process the file
        """
        _, ext = os.path.splitext(file_path)
        return ext.lower() in self.file_extensions

    def clone_repositories(self) -> List[str]:
        """
        Clone all repositories to the specified clone directory or temporary directories.

        :return: A list of paths to the cloned repositories.
        :raises: GitCommandError if cloning fails
        """
        cloned_paths = []
        for repo_url in self.repo_urls:
            try:
                repo_name = os.path.splitext(os.path.basename(repo_url))[0]
                if self.clone_dir:
                    os.makedirs(self.clone_dir, exist_ok=True)
                    repo_path = os.path.join(self.clone_dir, repo_name)
                else:
                    tmp_dir = tempfile.mkdtemp()
                    repo_path = os.path.join(tmp_dir, repo_name)

                if os.path.exists(repo_path):
                    logger.warning(f"Repository directory already exists: {repo_path}")
                    if os.path.isdir(os.path.join(repo_path, '.git')):
                        logger.info("Pulling latest changes instead of cloning")
                        repo = Repo(repo_path)
                        origin = repo.remotes.origin
                        origin.pull()
                    else:
                        raise GitCommandError("clone", f"Directory exists but is not a git repository: {repo_path}")
                else:
                    logger.info(f"Cloning repository {repo_url} into {repo_path}")
                    Repo.clone_from(repo_url, repo_path)

                cloned_paths.append(repo_path)
                self.cloned_repos.append(repo_path)

            except GitCommandError as e:
                logger.error(f"Failed to clone repository {repo_url}: {e}")
                raise

        return cloned_paths

    def load_documents(self) -> List[Document]:
        """
        Load documents from the cloned repositories with encoding detection.

        :return: A list of LangChain Document objects.
        :raises: Exception if document loading fails
        """
        try:
            cloned_paths = self.clone_repositories()
            all_documents = []

            for repo_path in cloned_paths:
                try:
                    logger.info(f"Loading documents from repository at {repo_path}")
                    
                    # Walk through the repository
                    for root, _, files in os.walk(repo_path):
                        for file in files:
                            file_path = os.path.join(root, file)
                            
                            # Skip files we don't want to process
                            if not self.should_process_file(file_path):
                                continue
                            
                            # Skip git directory
                            if '.git' in file_path:
                                continue

                            content = self._read_file_with_encoding(file_path)
                            if content is not None:
                                doc = Document(
                                    page_content=content,
                                    metadata={"source": file_path}
                                )
                                all_documents.append(doc)

                    logger.info(f"Loaded {len(all_documents)} documents from {repo_path}")
                
                except Exception as e:
                    logger.error(f"Failed to process repository at {repo_path}: {e}")
                    raise

            return all_documents

        except Exception as e:
            logger.error(f"Error in load_documents: {e}")
            self.cleanup()
            raise

    def load_documents_as_dicts(self) -> List[Dict[str, str]]:
        """
        Load documents and return them as a list of dictionaries containing source and content.

        :return: A list of dictionaries with 'source' and 'content' keys.
        :raises: Exception if document loading fails
        """
        try:
            docs = self.load_documents()
            document_list = []

            for doc in docs:
                source = doc.metadata.get('source', 'Unknown Source')
                content = doc.page_content if doc.page_content is not None else ''
                # document_list.append({
                #     'source': source,
                #     'content': content
                # })

                document_list.append(Document(
                    page_content=content,
                    metadata={'source': source}
                ))
            logger.info(f"Total documents prepared: {len(document_list)}")
            return document_list

        except Exception as e:
            logger.error(f"Error in load_documents_as_dicts: {e}")
            raise

    def cleanup(self):
        """
        Clean up cloned repositories if they were cloned to temporary directories.
        """
        for repo_path in self.cloned_repos:
            if not self.clone_dir and os.path.exists(repo_path):
                try:
                    logger.info(f"Removing temporary directory {repo_path}")
                    shutil.rmtree(repo_path)
                except Exception as e:
                    logger.error(f"Failed to remove temporary directory {repo_path}: {e}")

    def __enter__(self):
        """Context manager entry point."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit point."""
        self.cleanup()

    def __del__(self):
        """Destructor to ensure cleanup is called."""
        self.cleanup()
        



# Research Papers Document Loader
class ResearchPapersDocumentLoader:
    """
    A class to load research papers from arXiv, IEEE Xplore, and Elsevier APIs based on a search query.
    Returns the papers as a list of LangChain Document objects.
    """

    def __init__(
        self,
        ieee_api_key: Optional[str] = None,
        elsevier_api_key: Optional[str] = None,
    ):
        """
        Initialize the ResearchPapersDocumentLoader.

        :param ieee_api_key: API key for IEEE Xplore.
        :param elsevier_api_key: API key for Elsevier.
        """
        self.ieee_api_key = ieee_api_key or self._prompt_for_ieee_api_key()
        self.elsevier_api_key = elsevier_api_key or self._prompt_for_elsevier_api_key()

    def _prompt_for_ieee_api_key(self) -> str:
        """
        Prompt the user to enter their IEEE Xplore API Key securely.

        :return: The entered IEEE Xplore API Key.
        """
        return getpass("Enter your IEEE Xplore API Key: ")

    def _prompt_for_elsevier_api_key(self) -> str:
        """
        Prompt the user to enter their Elsevier API Key securely.

        :return: The entered Elsevier API Key.
        """
        return getpass("Enter your Elsevier API Key: ")

    def search_arxiv(self, query: str, max_results: int = 10) -> List[Document]:
        """
        Search and retrieve papers from arXiv based on a query.

        :param query: The search query string.
        :param max_results: Maximum number of papers to retrieve.
        :return: A list of LangChain Document objects.
        """
        logger.info(f"Searching arXiv for query: {query}")
        try:
            # Using the arxiv library to search
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            docs = []
            for result in search.results():
                content = f"Title: {result.title}\n\nAuthors: {', '.join([str(author) for author in result.authors])}\n\nSummary: {result.summary}"
                metadata = {
                    'source': result.entry_id,
                    'title': result.title,
                    'authors': ', '.join([str(author) for author in result.authors]),
                    'summary': result.summary,
                    'published': result.published.strftime("%Y-%m-%d"),
                    'url': result.pdf_url
                }
                doc = Document(page_content=content, metadata=metadata)
                docs.append(doc)
            logger.info(f"Retrieved {len(docs)} papers from arXiv.")
            return docs
        except Exception as e:
            logger.error(f"Failed to retrieve papers from arXiv: {e}")
            return []

    def search_ieee_xplore(self, query: str, max_results: int = 10) -> List[Document]:
        """
        Search and retrieve papers from IEEE Xplore based on a query.

        :param query: The search query string.
        :param max_results: Maximum number of papers to retrieve.
        :return: A list of LangChain Document objects.
        """
        logger.info(f"Searching IEEE Xplore for query: {query}")
        try:
            headers = {
                'X-API-Key': self.ieee_api_key,
            }
            params = {
                'querytext': query,
                'format': 'json',
                'max_records': max_results,
                'start_record': 1,
                'sort_field': 'relevance'
            }
            response = requests.get(
                'http://ieeexploreapi.ieee.org/api/v1/search/articles',
                headers=headers,
                params=params
            )
            if response.status_code != 200:
                logger.error(f"IEEE Xplore API Error {response.status_code}: {response.text}")
                return []
            data = response.json()
            documents = []
            for article in data.get('articles', []):
                title = article.get('article_title', 'No Title')
                authors = ', '.join([author.get('full_name', 'Unknown') for author in article.get('authors', [])])
                abstract = article.get('abstract', 'No Abstract')
                published_date = article.get('publication_date', 'No Date')
                url = article.get('ieee_url', 'No URL')
                content = f"Title: {title}\n\nAuthors: {authors}\n\nAbstract: {abstract}"
                metadata = {
                    'source': url,
                    'title': title,
                    'authors': authors,
                    'abstract': abstract,
                    'published': published_date,
                    'url': url
                }
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            logger.info(f"Retrieved {len(documents)} papers from IEEE Xplore.")
            return documents
        except Exception as e:
            logger.error(f"Failed to retrieve papers from IEEE Xplore: {e}")
            return []

    def search_elsevier(self, query: str, max_results: int = 10) -> List[Document]:
        """
        Search and retrieve papers from Elsevier's API based on a query.

        :param query: The search query string.
        :param max_results: Maximum number of papers to retrieve.
        :return: A list of LangChain Document objects.
        """
        logger.info(f"Searching Elsevier for query: {query}")
        try:
            headers = {
                'X-ELS-APIKey': self.elsevier_api_key,
                'Accept': 'application/json',
            }
            params = {
                'query': query,
                'count': max_results,
                'start': 0
            }
            response = requests.get(
                'https://api.elsevier.com/content/search/scopus',
                headers=headers,
                params=params
            )
            if response.status_code != 200:
                logger.error(f"Elsevier API Error {response.status_code}: {response.text}")
                return []
            data = response.json()
            documents = []
            for entry in data.get('search-results', {}).get('entry', []):
                title = entry.get('dc:title', 'No Title')
                authors = ', '.join([author.get('authname', 'Unknown') for author in entry.get('dc:creator', [])])
                abstract = entry.get('dc:description', 'No Abstract')
                published_date = entry.get('prism:coverDate', 'No Date')
                doi = entry.get('prism:doi', 'No DOI')
                url = f"https://doi.org/{doi}" if doi != 'No DOI' else 'No URL'
                content = f"Title: {title}\n\nAuthors: {authors}\n\nAbstract: {abstract}"
                metadata = {
                    'source': url,
                    'title': title,
                    'authors': authors,
                    'abstract': abstract,
                    'published': published_date,
                    'doi': doi,
                    'url': url
                }
                doc = Document(page_content=content, metadata=metadata)
                documents.append(doc)
            logger.info(f"Retrieved {len(documents)} papers from Elsevier.")
            return documents
        except Exception as e:
            logger.error(f"Failed to retrieve papers from Elsevier: {e}")
            return []

    def load_papers(
        self,
        query: str,
        max_results: int = 10,
        sources: Optional[List[str]] = None
    ) -> List[Document]:
        """
        Load research papers from specified sources based on a query.

        :param query: The search query string.
        :param max_results: Maximum number of papers to retrieve per source.
        :param sources: List of sources to search in. Options: ['arxiv', 'ieee', 'elsevier']. If None, searches all.
        :return: A combined list of LangChain Document objects.
        """
        if sources is None:
            sources = ['arxiv', 'ieee', 'elsevier']
        
        all_documents = []
        
        if 'arxiv' in sources:
            arxiv_docs = self.search_arxiv(query, max_results)
            all_documents.extend(arxiv_docs)
        
        if 'ieee' in sources:
            ieee_docs = self.search_ieee_xplore(query, max_results)
            all_documents.extend(ieee_docs)
        
        if 'elsevier' in sources:
            elsevier_docs = self.search_elsevier(query, max_results)
            all_documents.extend(elsevier_docs)
        
        logger.info(f"Total papers retrieved from all sources: {len(all_documents)}")
        return all_documents

    def load_papers_as_dicts(
        self,
        query: str,
        max_results: int = 10,
        sources: Optional[List[str]] = None
    ) -> List[Dict[str, str]]:
        """
        Load papers and return them as a list of dictionaries containing source and content.

        :param query: The search query string.
        :param max_results: Maximum number of papers to retrieve per source.
        :param sources: List of sources to search in. Options: ['arxiv', 'ieee', 'elsevier']. If None, searches all.
        :return: A list of dictionaries with 'source' and 'content' keys.
        """
        docs = self.load_papers(query, max_results, sources)
        document_list = []
        
        for doc in docs:
            source = doc.metadata.get('source', 'Unknown Source')
            content = doc.page_content or ''
            # document_list.append({
            #     'source': source,
            #     'content': content
            # })
            document_list.append(Document(
                page_content=content,
                metadata={'source': source}
            ))
            
        logger.info(f"Total documents prepared: {len(document_list)}")
        return document_list





