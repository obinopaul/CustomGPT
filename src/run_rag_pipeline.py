#!/usr/bin/env python3
"""
run_rag_pipeline.py

An advanced, flexible script/class to orchestrate:
 - Data validation
 - Text splitting from various tasks (file, url, GitHub repo, GitHub issues, research papers)
 - Vector store creation (Chroma or Pinecone)
 - LLM pipeline setup (OpenAI, HuggingFace, Llama)
 - Retrieval-Augmented Generation (RAG) or basic chat

Usage Options:
1) Import as a module:
    from run_rag_pipeline import RunChatbot

    chatbot = RunChatbot(
        model_type="openai",
        api_key="YOUR_OPENAI_API_KEY",
        use_rag=True,
        vector_store_type="pinecone",
        embedding_type="openai",
        data_task="github_repo",
        data_value="https://github.com/obinopaul/licence-plate-detection.git",
        ...
    )
    chatbot.setup_data()            # validate and split
    chatbot.setup_vector_store()    # build & populate the vector store
    chatbot.setup_llm_pipeline()    # create the pipeline
    answer = chatbot.chat("What is the capital of France?")

2) Command line usage (sketch):
    python run_rag_pipeline.py --model_type openai --api_key sk-ABC123 --use_rag True ...
   (Implement arg parsing if you want a CLI interface.)

Author: Your Name / Date
"""

import os
import logging
from typing import Optional, Dict, Any, List

# These imports rely on your existing folder structure.
# Adjust them to match your actual project if needed.
from src.data_validate import DataValidator
from pydantic import ValidationError
from src.rag.text_splitter import TextSplitter
from src.rag.vector_store import VectorStoreManager
from src.llm.llm_pipeline import LLMPipeline
from langchain.docstore.document import Document

# -----------------------------------------------------------------------------
# Logging
# -----------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# RunChatbot Class
# -----------------------------------------------------------------------------
class RunChatbot:
    """
    A single advanced class to orchestrate:
      - Data validation
      - Text splitting (various tasks: 'file', 'url', 'github_repo', 'github_issues', 'research_papers')
      - Building a vector store if use_rag=True
      - Building an LLM pipeline (OpenAI, HuggingFace, Llama)
      - Answering queries with or without RAG
    """

    def __init__(
        self,
        model_type: str,
        api_key: Optional[str] = None,
        use_rag: bool = False,
        data_task: Optional[str] = None,        # "file", "url", "github_repo", "github_issues", "research_papers". this refers to the specific task for rag
        data_value: Optional[str] = None,       # Path, URL, etc.   This
        vector_store_type: Optional[str] = "pinecone",   # "pinecone" or "chroma"
        embedding_type: Optional[str] = "openai",       # "openai", "huggingface", "llama"
        model_path: Optional[str] = None,           # for Llama
        model_name: Optional[str] = None,           # for HuggingFace
        model:Optional[str] = None,                 # for OpenAI model. This is the model name
        temperature: float = 0.7,
        # max_tokens: int = 150,
        top_p: float = 0.9,
        freq_penalty: float = 0.3,
        pres_penalty: float = 0.5,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
        elsevier_api_key: Optional[str] = None,
        ieee_api_key: Optional[str] = None,
        
        **kwargs
    ):
        """
        :param model_type:         "openai", "huggingface", or "llama"
        :param api_key:            For OpenAI or private HuggingFace
        :param use_rag:            If True, set up a vector store and do retrieval.
        :param data_task:          One of ["file", "url", "github_repo", "github_issues", "research_papers"] or None
        :param data_value:         The data source path/URL/etc. you want to process.
        :param vector_store_type:  "pinecone" or "chroma"
        :param embedding_type:     "openai", "huggingface", "ollama"  (Used if use_rag=True)
        :param model_path:         If ollama model is local, pass path. (ollama only)
        :param model_name:         For huggingface. (huggingface only)
        :param temperature:        LLM temperature
        :param max_tokens:         LLM max tokens
        :param top_p:              LLM top_p
        :param freq_penalty:       LLM frequency penalty
        :param pres_penalty:       LLM presence penalty
        :param chunk_size:         For text splitting
        :param chunk_overlap:      For text splitting
        :param kwargs:             Additional arguments if needed
        """
        self.model_type = model_type
        self.api_key = api_key
        self.use_rag = use_rag
        self.data_task = data_task
        self.data_value = data_value
        self.vector_store_type = vector_store_type
        self.embedding_type = embedding_type
        self.model_path = model_path
        self.model_name = model_name
        self.ieee_api_key = ieee_api_key
        self.elsevier_api_key = elsevier_api_key

        self.temperature = temperature
        # self.max_tokens = max_tokens
        self.top_p = top_p
        self.freq_penalty = freq_penalty
        self.pres_penalty = pres_penalty
        self.model = model

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        # We'll create or store references to your sub-components:
        self.validator = None
        self.splitter = None
        self.vector_store_manager = None
        self.pipeline = None

        self.docs: List[Document] = []   # Store splitted docs
        self.retriever = None            # Will store the created retriever if use_rag

        # Additional
        self.kwargs = kwargs  # For advanced usage

    # -------------------------------------------------------------------------
    # 1) Setup Data (Validate + Split)
    # -------------------------------------------------------------------------
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

    # -------------------------------------------------------------------------
    # 2) Setup Vector Store
    # -------------------------------------------------------------------------
    def setup_vector_store(self) -> None:
        """
        If use_rag=True, build a VectorStoreManager (Chroma or Pinecone),
        add documents, and create a retriever.
        """
        logger.info("=== [RunChatbot] setup_vector_store ===")
        if not self.use_rag:
            logger.info("RAG not enabled (use_rag=False). No vector store created.")
            return

        if not self.vector_store_type:
            raise ValueError("No vector_store_type given (e.g., 'pinecone' or 'chroma').")

        if not self.embedding_type:
            raise ValueError("No embedding_type given (e.g., 'openai', 'huggingface', 'llama').")

        # 1) Build the vector store manager
        self.vector_store_manager = VectorStoreManager(
            vector_store_type=self.vector_store_type,
            embedding_type=self.embedding_type,
            pinecone_api_key=self.kwargs.get("pinecone_api_key", os.getenv("PINECONE_API_KEY")),
            pinecone_index_name=self.kwargs.get("pinecone_index_name", "langchain-index"),
            collection_name=self.kwargs.get("collection_name", "langchain_collection"),
            embedding_model_name = "text-embedding-ada-002" if self.embedding_type == "openai" else self.model_name
        )

        # 2) Add docs
        if self.docs:
            self.vector_store_manager.add_documents(documents=self.docs, clear_store=True)
            logger.info("Documents added to VectorStoreManager.")

        # 3) Create a retriever
        self.retriever = self.vector_store_manager.as_retriever()
        logger.info("Retriever created successfully.")

    # -------------------------------------------------------------------------
    # 3) Setup LLM Pipeline
    # -------------------------------------------------------------------------
    def setup_llm_pipeline(self) -> None:
        """
        Creates an LLMPipeline from your src.llm.llm_pipeline, using
        model_type = openai/huggingface/llama, plus the relevant arguments.
        """
        logger.info("=== [RunChatbot] setup_llm_pipeline ===")
        self.pipeline = LLMPipeline(
            model_type=self.model_type,
            api_key=self.api_key,
            model_path=self.model_path,
            model_name=self.model_name,
            memory_type="buffer",    # or "window" if you want
            temperature=self.temperature,
            window_size=5,
            return_messages=True,
            model = self.model,
            k=4
            # You can pass other kwargs if your pipeline or chatbot requires it
        )
        logger.info("LLMPipeline created successfully.")

        if self.use_rag and self.retriever:
            # Build retrieval QA chain
            self.pipeline.chatbot.create_retrieval_qa_chain(retriever=self.retriever)
            logger.info("Retrieval QA chain established in the pipeline.")

    # -------------------------------------------------------------------------
    # 4) Chat / RAG
    # -------------------------------------------------------------------------
    def chat(self, query: str, with_sources: bool = True) -> str:
        """
        Single method to get an answer. If use_rag=True, uses the retrieval
        chain. If not, does a normal chat. If with_sources=True and RAG is on,
        returns sources as well.
        """
        logger.info(f"=== [RunChatbot] chat - query: {query} ===")

        if not self.pipeline:
            raise ValueError("Pipeline not set up. Call setup_llm_pipeline() first.")

        # If RAG is enabled, we have a chain
        if self.use_rag and self.retriever:
            if with_sources:
                res_with_src = self.pipeline.generate_response_with_sources(query, self.retriever)
                # Format an output or just return dict
                answer = res_with_src["result"]
                sources = res_with_src["sources"]
                # Build a combined string or return
                source_list = [doc.metadata.get("source", "(No source)") for doc in sources]
                return {
                    "result": answer,
                    "sources": source_list
                }
            else:
                # RAG without sources
                res = self.pipeline.generate_response(query, self.retriever)
                return {
                    "result": res.get("result", "")
                }
                
        elif self.use_rag and not self.retriever:
            # Normal chat (no RAG)
            # We can call pipeline.run(...) or pipeline.chat(...) depending on your design
            output = self.pipeline.run(query) 
            return {
                "result": output.get("response", "")
            }  # pipeline.run returns {"query","response"}

        else:
            # Normal chat (no RAG)
            # We can call pipeline.run(...) or pipeline.chat(...) depending on your design
            output = self.pipeline.run(query) 
            return {
                "result": output.get("response", "")
            }  # pipeline.run returns {"query","response"}
# -------------------------------------------------------------------------
# Optional: If you want a direct CLI usage or a main():
# -------------------------------------------------------------------------
def main():
    """Example usage from the command line or direct call."""
    # Example: create a chatbot that uses RAG with Pinecone + OpenAI
    chatbot = RunChatbot(
        model_type="openai",
        api_key="YOUR_OPENAI_API_KEY",
        use_rag=True,
        data_task="file",  # or "url", "github_repo", etc.
        data_value="./data",  # path to the folder
        vector_store_type="pinecone",
        embedding_type="openai",
        temperature=0.7
    )

    # 1) Validate & Split
    chatbot.setup_data()
    # 2) Build VectorStore, add docs, get retriever
    chatbot.setup_vector_store()
    # 3) Create LLM pipeline, build chain
    chatbot.setup_llm_pipeline()

    # 4) Chat
    query = "What's the capital of France?"
    result = chatbot.chat(query, with_sources=True)
    print(result)

if __name__ == "__main__":
    main()
