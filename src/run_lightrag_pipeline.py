#!/usr/bin/env python3
"""
run_lightrag_pipeline.py

A class to orchestrate:
 - Text splitting from various tasks (file, URL, GitHub repo, GitHub issues, research papers)
 - LightRAG-based vector store creation and querying
 - Integration with multiple LLM models (OpenAI, HuggingFace, Ollama)
 - Retrieval-Augmented Generation (RAG)

Usage:
1) Import as a module:
    from run_lightrag_pipeline import RunLightRAGChatbot

    chatbot = RunLightRAGChatbot(
        llm_model_func=gpt_4o_mini_complete,
        working_dir="./my_working_dir",
        data_task="file",
        data_value="./data/my_file.pdf",
        chunk_size=1000,
        chunk_overlap=200
    )
    chatbot.setup_data()
    chatbot.setup_lightrag()
    answer = chatbot.query("What is the capital of France?")

2) Command line usage (sketch):
    python run_lightrag_pipeline.py --llm_model_func gpt_4o_mini_complete ...
   (Implement arg parsing if needed.)
"""

import os
import shutil
import logging
from typing import Optional, Dict, Any, List
from lightrag import LightRAG, QueryParam
from lightrag.llm import gpt_4o_mini_complete
from lightrag import LightRAG, QueryParam
from lightrag.utils import EmbeddingFunc
from lightrag.llm import hf_model_complete, hf_embedding
from lightrag.llm import ollama_embedding, ollama_model_complete
from lightrag.llm import openai_complete_if_cache, openai_embedding
from transformers import AutoModel, AutoTokenizer
import asyncio
# import nest_asyncio
# import aiofiles

# Other imports
from src.rag.text_splitter import TextSplitter
from langchain.docstore.document import Document

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -------------------------------------------------------------------------
# RunLightRAGChatbot Class
# -------------------------------------------------------------------------
class RunLightRAGChatbot:
    """
    A class to orchestrate:
      - Text splitting (file, URL, GitHub repo, GitHub issues, research papers)
      - LightRAG-based vector store and querying
      - Integration with LLMs
    """

    def __init__(
        self,
        model_type: str,
        working_dir: str = "./lightrag_working_dir",
        openai_model: Optional[str] = None,   # example is gpt_4o_mini_complete
        data_task: Optional[str] = None,
        data_value: Optional[str] = None,
        openai_api_key: Optional[str] = None,
        openai_embedding_model: Optional[str] = None,  # example is "text-embedding-3-large"
        huggingface_model_name: Optional[str] = None,
        huggingface_tokenizer_name: Optional[str] = None,
        ollama_embedding_model: Optional[str] = None,  # example is "nomic-embed-text"
        ollama_model_name: Optional[str] = None,    # example is "llama_3.2"
        ollama_host: Optional[str] = None,  # example is http://localhost:11434
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        max_token_size: Optional[int] = None,
        **kwargs
    ):
        """
        :param model_type: Type of model ("openai", "huggingface", "ollama")
        :param working_dir: Working directory for LightRAG
        :param llm_model_name: Model name (for OpenAI, HuggingFace, or Ollama)
        :param data_task: Task type ('file', 'url', 'github_repo', etc.)
        :param data_value: Input source (file path, URL, etc.)
        :param openai_api_key: API key for OpenAI
        :param huggingface_model_name: HuggingFace model name
        :param huggingface_tokenizer_name: HuggingFace tokenizer name
        :param ollama_host: Host for Ollama models
        :param chunk_size: Text chunk size
        :param chunk_overlap: Text chunk overlap
        """
        self.model_type = model_type.lower()
        self.working_dir = working_dir
        self.openai_model = openai_model
        self.data_task = data_task
        self.data_value = data_value
        self.openai_api_key = openai_api_key if openai_api_key else os.getenv("OPENAI_API_KEY")
        self.openai_embedding_model = openai_embedding_model if openai_embedding_model else "text-embedding-3-large"
        self.huggingface_model_name = huggingface_model_name
        self.huggingface_tokenizer_name = huggingface_tokenizer_name
        self.ollama_embedding_model = ollama_embedding_model
        self.ollama_model_name = ollama_model_name
        self.ollama_host = ollama_host
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.max_token_size = max_token_size
        self.docs: List[Document] = []
        self.rag = None

        # Additional
        self.kwargs = kwargs  # For advanced usage

        # Ensure the working directory exists
        os.makedirs(self.working_dir, exist_ok=True)
        
    def setup_data(self) -> None:
        """
        Validate data_type + data_value if needed, then use TextSplitter to create doc chunks.
        For example: "file", "url", "github_repo", "github_issues", "research_papers", or raw "text".
        """
        logger.info("=== [RunChatbot] setup_data ===")
        
        # 1) Validate data (Optional)
        # self.validator = DataValidator()
        # if self.data_task and self.data_value:
        #     try:
        #         validated = self.validator.validate_input(
        #             data_type=self.data_task,  # e.g. "github_repo_url"
        #             data=self.data_value
        #         )
        #         logger.info(f"Data validated: {validated}")
        #     except ValidationError as e:
        #         logger.warning(f"Validation error: {e}")
        # else:
        #     logger.warning("No data task or data value provided, skipping validation.")

        # 2) Split
        self.splitter = TextSplitter(
            splitter_type='recursive',
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

        # We'll handle each data_task
        # Handle each data_task
        if self.data_task == "file":
            if isinstance(self.data_value, list):
                # If it's a list of file paths
                file_paths = [path for path in self.data_value if os.path.isfile(path)]
                if not file_paths:
                    raise ValueError("No valid file paths provided in data_value.")
                self.docs = self.splitter.split_file_documents(file_paths)
            elif os.path.isfile(self.data_value):
                # If it's a single file path
                self.docs = self.splitter.split_file_documents([self.data_value])
            else:
                raise ValueError("Invalid data_value for 'file' data_task. Must be a file path or list of file paths.")

                    

        elif self.data_task == "url":
            self.docs = self.splitter.split_url_documents(self.data_value)

        elif self.data_task == "github_repo":
            # data_value might be a single URL or a list
            repo_urls = [self.data_value] if isinstance(self.data_value, str) else self.data_value
            clone_dir = self.kwargs.get("clone_dir", "./cloned_repo")
            self.docs = self.splitter.split_github_repo_documents(repo_urls=repo_urls, clone_dir=clone_dir)

        elif self.data_task == "github_issues":
            # data_value might be a single "owner/repo" or multiple
            repos = [self.data_value] if isinstance(self.data_value, str) else self.data_value
            access_token = self.kwargs.get("github_access_token", "")
            self.docs = self.splitter.split_github_issues_documents(repos=repos, access_token=access_token)

        elif self.data_task == "research_papers":
            # data_value might be a query string
            query_str = self.data_value
            max_results = self.kwargs.get("max_results", 20)
            sources = self.kwargs.get("sources", ["ieee", "elsevier", "arxiv"])
            # ieee_api_key = self.kwargs.get("ieee_api_key", "")
            # elsevier_api_key = self.kwargs.get("elsevier_api_key", "")
            self.docs = self.splitter.split_research_papers_documents(
                query=query_str,
                max_results=max_results,
                sources=sources,
                ieee_api_key = self.ieee_api_key,
                elsevier_api_key = self.elsevier_api_key
            )

        # elif self.data_task == "text":
        #     # If the data_value is raw text
        #     # or you could store them in a Document object
        #     text_docs = [Document(page_content=self.data_value)]
        #     self.docs = self.splitter.split_documents(text_docs)

        else:
            logger.info("No recognized data_task. If you only want a normal chat with no data ingestion, skip this.")
            self.docs = []

        logger.info(f"Total splitted docs: {len(self.docs)}")

    def setup_lightrag(self) -> None:
        """
        Initialize LightRAG with the selected LLM model and documents.
        """
        logger.info("=== [RunLightRAGChatbot] setup_lightrag ===")

        # Initialize the LLM function based on model type
        if self.model_type == "openai":
            if not self.openai_api_key or not self.openai_model:
                raise ValueError("OpenAI API key and model name are required for OpenAI models.")

            # Define asynchronous embedding and LLM functions
            async def openai_llm_func(prompt, **kwargs):
                return await openai_complete_if_cache(
                    model=self.openai_model, prompt=prompt, api_key=self.openai_api_key, **kwargs
                )

            async def openai_embedding_func(texts: list[str]):
                return await openai_embedding(
                    texts,
                    model="text-embedding-3-large",
                    api_key=self.openai_api_key,
                )

            # Dynamically fetch the embedding dimension
            if not hasattr(self, "_embedding_dim"):
                async def get_embedding_dim():
                    test_text = ["This is a test sentence."]
                    embedding = await openai_embedding_func(test_text)
                    return embedding.shape[1]

                self._embedding_dim = asyncio.run(get_embedding_dim())
                
            self._embedding_dim = asyncio.run(get_embedding_dim())
            # # Get embedding dimensions dynamically
            # async def get_embedding_dim():
            #     test_text = ["This is a test sentence."]
            #     embedding = await openai_embedding_func(test_text)
            #     return embedding.shape[1]

            # Check if NanoVectorDB needs resetting
            existing_embedding_dim_path = os.path.join(self.working_dir, "embedding_dim.txt")
            if os.path.exists(self.working_dir):
                if os.path.exists(existing_embedding_dim_path):
                    with open(existing_embedding_dim_path, "r") as f:
                        existing_dim = int(f.read().strip())
                    if existing_dim != self._embedding_dim:
                        logger.warning(f"Embedding dimension mismatch: clearing NanoVectorDB (expected {existing_dim}, got {self._embedding_dim}).")
                        shutil.rmtree(self.working_dir)
                else:
                    logger.warning("Embedding dimension file not found. Resetting NanoVectorDB.")
                    shutil.rmtree(self.working_dir)

            # Save the current embedding dimension for future reference
            os.makedirs(self.working_dir, exist_ok=True)
            with open(existing_embedding_dim_path, "w") as f:
                f.write(str(self._embedding_dim))
                
            # Initialize LightRAG for OpenAI
            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=openai_llm_func,
                embedding_func=EmbeddingFunc(
                    embedding_dim=self._embedding_dim,
                    max_token_size=self.max_token_size or 8192,
                    func=openai_embedding_func,
                ),
            )
        
        # Initialize the LLM function based on model type
        if self.model_type == "openai":
            if not self.openai_api_key or not self.openai_model:
                raise ValueError("OpenAI API key and model name are required for OpenAI models.")
            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=lambda prompt, **kwargs: openai_complete_if_cache(
                    model=self.openai_model, prompt=prompt, api_key=self.openai_api_key, **kwargs
                ),
            )

        elif self.model_type == "huggingface":
            if not self.huggingface_model_name:
                raise ValueError("HuggingFace model name is required.")
            if not self.huggingface_tokenizer_name:
                self.huggingface_tokenizer_name = self.huggingface_model_name  # Default to model name

            tokenizer = AutoTokenizer.from_pretrained(self.huggingface_tokenizer_name)
            model = AutoModel.from_pretrained(self.huggingface_model_name)

            embedding_func = EmbeddingFunc(
                embedding_dim=384,  # Update if your embedding model's dimension differs
                max_token_size=5000,  # Adjust based on your model's token limit
                func=lambda texts: hf_embedding(texts, tokenizer=tokenizer, embed_model=model),
            )

            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=lambda prompt, **kwargs: hf_model_complete(
                    prompt, model=model, tokenizer=tokenizer, **kwargs
                ),
                embedding_func=embedding_func,
            )
    

        elif self.model_type == "ollama":
            if not self.ollama_model_name:
                raise ValueError("Ollama model name is required.")
            if not self.host:
                raise ValueError("Ollama model host is required.")

            # Configure the embedding function
            embedding_func = EmbeddingFunc(
                embedding_dim=768,  # Default dimension for nomic-embed-text
                max_token_size=8192,  # Adjust based on your requirements
                func=lambda texts: ollama_embedding(
                    texts,
                    embed_model= self.ollama_embedding_model if self.ollama_embedding_model else "nomic-embed-text",  # Use default or user-specified
                    host= self.ollama_host if self.ollama_host else "http://localhost:11434",
                ),
            )

            # Initialize LightRAG for Ollama
            self.rag = LightRAG(
                working_dir=self.working_dir,
                llm_model_func=lambda prompt, **kwargs: ollama_model_complete(
                    prompt,
                    model=self.ollama_model_name,
                    host= self.ollama_host if self.ollama_host else "http://localhost:11434"
                    **kwargs,
                ),
                llm_model_name=self.ollama_model_name,
                llm_model_max_async=self.max_async or 4,
                llm_model_max_token_size=self.max_token_size or 8192,
                llm_model_kwargs={"host": self.ollama_host if self.ollama_host else "http://localhost:11434", "options": {"num_ctx": self.max_token_size or 8192}},
                embedding_func=embedding_func,
            )


        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")
        
        
        # # Insert documents into LightRAG
        # for doc in self.docs:
        #     self.rag.insert(doc.page_content)
        # logger.info("Documents inserted into LightRAG successfully.")

        if not self.docs:
            logger.warning("No documents found to insert into LightRAG.")
            return
    
        # Insert documents into LightRAG
        for doc in self.docs:
            if isinstance(doc, Document) and hasattr(doc, "page_content"):
                content = doc.page_content.strip()
            elif isinstance(doc, str):
                content = doc.strip()
            else:
                logger.warning(f"Skipping unsupported document format: {type(doc)}")
                continue

            if content:
                try:
                    self.rag.insert(content)
                except Exception as e:
                    logger.error(f"Error inserting document into LightRAG: {e}")
            
    def chat(self, prompt: str, mode: str = "naive") -> str:
        """
        Query the LightRAG system.
        :param prompt: The input question or query
        :param mode: Query mode ('naive', 'local', 'global', 'hybrid')
        :return: Answer string
        """
        if not self.rag:
            raise ValueError("LightRAG is not set up. Call setup_lightrag() first.")

        logger.info(f"Querying LightRAG with prompt: {prompt}")
        result = self.rag.query(prompt, param=QueryParam(mode=mode))
        return result


# -------------------------------------------------------------------------
# Main for testing
# -------------------------------------------------------------------------
if __name__ == "__main__":
    chatbot = RunLightRAGChatbot(
        llm_model_func=gpt_4o_mini_complete,
        data_task="file",
        data_value="./sample.pdf",
    )
    chatbot.setup_data()
    chatbot.setup_lightrag()
    answer = chatbot.query("What are the key points in this document?")
    print(answer)
