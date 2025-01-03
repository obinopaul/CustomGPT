# src/rag/vector_store.py

import os
import logging
from typing import List, Union, Optional, Dict, Any
from uuid import uuid4
from getpass import getpass

# Import Vector Stores
from langchain_pinecone import PineconeVectorStore
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OpenAIEmbeddings, LlamaCppEmbeddings
from langchain_ollama import OllamaEmbeddings  # Ensure langchain-ollama is installed
from langchain_huggingface import HuggingFaceEmbeddings  # Ensure langchain-huggingface is installed
from langchain.schema import BaseRetriever

# Pydantic
from pydantic import PrivateAttr

# Import Pinecone Client
import pinecone
from pinecone import Pinecone, ServerlessSpec
import chromadb

# Import LangChain Document
from langchain.docstore.document import Document


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



class VectorStoreManager:
    """
    A class to manage vector stores and handle embedding and storing of documents from various sources.
    Supports Chroma and Pinecone as vector stores, and OpenAI, Llama, and HuggingFace as embedding models.
    """
    def __init__(
        self,
        vector_store_type: str = "pinecone",  # Options: 'chroma', 'pinecone'
        embedding_type: Optional[str] = None,     # Options: 'openai', 'llama', 'huggingface'
        embedding_model_name: Optional[str] = None,  # Required for 'llama' and 'huggingface'
        model_name_for_token_splitter: Optional[str] = None,  # Required for 'token' splitter with OpenAI
        persist_directory: Optional[str] = "./chroma_db",   # For Chroma
        pinecone_api_key: Optional[str] = None,             # For Pinecone
        # pinecone_environment: Optional[str] = None,         # For Pinecone
        pinecone_index_name: str = "langchain-index",       # For Pinecone
        collection_name: str = "langchain_collection",      # For Chroma
        **kwargs
    ):
        """
        Initialize the VectorStoreManager.
        :param vector_store_type: Type of vector store to use ('chroma' or 'pinecone').
        :param embedding_type: Type of embedding model to use ('openai', 'llama', 'huggingface').
        :param embedding_model_name: Model name for 'llama' and 'huggingface' embeddings.
        :param model_name_for_token_splitter: Model name for TokenTextSplitter (required for 'openai' with token splitting).
        :param persist_directory: Directory to persist Chroma vector store data.
        :param pinecone_api_key: API key for Pinecone.
        # :param pinecone_environment: Environment for Pinecone (e.g., 'us-east1-gcp').
        :param pinecone_index_name: Name of the Pinecone index.
        :param collection_name: Name of the Chroma collection.
        :param kwargs: Additional keyword arguments.
        """
        self.vector_store_type = vector_store_type.lower()
        self.embedding_type = embedding_type.lower() if embedding_type else None
        self.embedding_model_name = embedding_model_name
        self.model_name_for_token_splitter = model_name_for_token_splitter
        self.persist_directory = persist_directory
        self.pinecone_api_key = pinecone_api_key or os.getenv("PINECONE_API_KEY")
        # self.pinecone_environment = pinecone_environment or os.getenv("PINECONE_ENVIRONMENT")
        self.pinecone_index_name = pinecone_index_name
        self.collection_name = collection_name
        self.vector_store = None
        self.embeddings = None
        self.dimension = None  # We’ll set this once we embed a dummy query
        
        # Initialize Embeddings
        self._initialize_embeddings()
        # Initialize Vector Store
        self._initialize_vector_store()
        
    def _initialize_embeddings(self):
        """
        Initialize the embedding model based on the selected embedding_type.
        """
        logger.info(f"Initializing embeddings with type: {self.embedding_type}")
        if self.vector_store_type == "pinecone":
            if self.embedding_type != "openai":
                raise ValueError("Pinecone vector store only supports OpenAI embeddings.")
            openai_api_key = os.getenv("OPENAI_API_KEY") or getpass("Enter your OpenAI API Key: ")
            self.embeddings = OpenAIEmbeddings(
                model="text-embedding-ada-002",  # You can make this configurable
                # disallowed_search=(),
                openai_api_key=openai_api_key
            )
        elif self.vector_store_type == "chroma":
            if self.embedding_type == "openai":
                openai_api_key = os.getenv("OPENAI_API_KEY") or getpass("Enter your OpenAI API Key: ")
                self.embeddings = OpenAIEmbeddings(
                    model="text-embedding-ada-002",  # You can make this configurable
                    # disallowed_search=(),
                    openai_api_key=openai_api_key
                )
                dummy_vector = self.embeddings.embed_query("test dimension")
                self.dimension = len(dummy_vector)
            elif self.embedding_type == "llama":
                if not self.embedding_model_name:
                    raise ValueError("embedding_model_name must be provided for Llama embeddings.")
                self.embeddings = OllamaEmbeddings(
                    model=self.embedding_model_name  # e.g., "llama-7b"
                )
            elif self.embedding_type == "huggingface":
                if not self.embedding_model_name:
                    raise ValueError("embedding_model_name must be provided for HuggingFace embeddings.")
                self.embeddings = HuggingFaceEmbeddings(
                    model_name=self.embedding_model_name,  # e.g., "sentence-transformers/all-mpnet-base-v2"
                    model_kwargs={'device': 'cpu'},         # Change to 'cuda' if GPU is available
                    encode_kwargs={'normalize_embeddings': False}
                )
            else:
                raise ValueError(f"Unsupported embedding_type: {self.embedding_type}")
        else:
            raise ValueError(f"Unsupported vector_store_type: {self.vector_store_type}")
        
        # Now compute dimension by embedding a dummy query
        dummy_vector = self.embeddings.embed_query("test dimension")
        self.dimension = len(dummy_vector)
        logger.info("Embeddings initialized successfully.")
        
    def _initialize_vector_store(self):
        logger.info(f"Initializing vector store with type: {self.vector_store_type}")
        if self.vector_store_type == "chroma":
            self._initialize_chroma_vector_store()
        elif self.vector_store_type == "pinecone":
            self._initialize_pinecone_vector_store()
        else:
            raise ValueError(f"Unsupported vector_store_type: {self.vector_store_type}")
        logger.info("Vector store initialized successfully.")

    def _initialize_chroma_vector_store(self):
        self.vector_store = Chroma(
            collection_name=self.collection_name,
            embedding_function=self.embeddings,
            persist_directory=self.persist_directory
        )
        logger.info(f"Chroma vector store initialized with collection '{self.collection_name}' and persist directory '{self.persist_directory}'.")



    def _initialize_pinecone_vector_store(self):
        """
        For Pinecone usage. 
        Checks if the index exists; if it does, deletes and recreates it; otherwise, creates it from scratch.
        """
        if not self.pinecone_api_key:
            raise ValueError("Pinecone API key must be provided for Pinecone vector store.")

        # Instantiate a Pinecone client
        pc = Pinecone(pinecone_api_key=self.pinecone_api_key)
        existing_indexes = [index_info["name"] for index_info in pc.list_indexes()]
        
        try:
            # Check if the index exists
            if self.pinecone_index_name in existing_indexes:
                logger.info(f"Pinecone index '{self.pinecone_index_name}' exists. Deleting it.")
                pc.delete_index(self.pinecone_index_name)
                logger.info(f"Index '{self.pinecone_index_name}' deleted successfully.")
            
            # Create the index
            logger.info(f"Creating Pinecone index '{self.pinecone_index_name}'.")
            pc.create_index(
                name=self.pinecone_index_name,
                dimension=self.dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1")
            )
            logger.info(f"Pinecone index '{self.pinecone_index_name}' created successfully.")
            # self._wait_for_pinecone_index_to_be_ready()
            
            # Connect to the index
            self.index = pc.Index(self.pinecone_index_name)
            self.vector_store = PineconeVectorStore(index=self.index, embedding=self.embeddings)
            logger.info(f"Pinecone vector store initialized with index '{self.pinecone_index_name}'.")
        
        except Exception as e:
            logger.error(f"An error occurred while initializing the Pinecone vector store: {e}")
            raise



    def _wait_for_pinecone_index_to_be_ready(self):
        while True:
            try:
                index_info = pinecone.describe_index(self.pinecone_index_name)
                if index_info.status == "ready":
                    break
            except Exception:
                pass

    # -------------------------------------------------------------------------
    # Public Methods
    # -------------------------------------------------------------------------
                   
    def add_documents(
        self,
        documents: List[Document],
        ids: Optional[List[str]] = None,
        clear_store: bool = False
    ):
        """
        Add a list of Document objects to the vector store.
        :param documents: List of LangChain Document objects.
        :param ids: Optional list of unique identifiers for the documents.
        :param clear_store: If True, clears the vector store before adding documents.
        """
        if clear_store:
            logger.info("Clearing vector store before adding new documents.")
            self.clear_vector_store()
        
        if not documents:
            logger.warning("No documents provided to add to the vector store.")
            return
        if not ids:
            ids = [str(uuid4()) for _ in range(len(documents))]
        logger.info(f"Adding {len(documents)} documents to the vector store.")
        
        try:
            self.vector_store.add_documents(documents=documents, ids=ids)
            logger.info("Documents added to the vector store successfully.")
        except Exception as e:
            logger.error(f"Failed to add documents to the vector store: {e}") 
            
    def search(
        self,
        query: str,
        top_k: int = 10
    ) -> List[Dict]:
        """
        Perform a similarity search on the vector store.
        :param query: The search query string.
        :param top_k: Number of top similar documents to retrieve.
        :return: List of dictionaries containing 'score' and 'document' keys.
        """
        logger.info(f"Performing similarity search for query: '{query}' with top_k: {top_k}")
        try:
            results = self.vector_store.similarity_search_with_score(query, k=top_k)
            formatted_results = []
            for doc, score in results:
                formatted_results.append({
                    'score': score,
                    'document': doc.page_content,
                    'metadata': doc.metadata
                })
            logger.info(f"Retrieved {len(formatted_results)} similar documents.")
            return formatted_results
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {e}")
            return []
        
    def persist(self):
        """
        Persist the vector store to disk (applicable for Chroma).
        """
        if self.vector_store_type == "chroma":
            logger.info(f"Persisting Chroma vector store to '{self.persist_directory}'.")
            self.vector_store.persist()
            logger.info("Chroma vector store persisted successfully.")
        else:
            logger.info("Persist operation is not applicable for Pinecone vector store.")
          
            
    def as_retriever(
        self,
        search_type: str = "similarity",
        search_kwargs: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> BaseRetriever:
        """
        Returns a BaseRetriever-compatible object so that LLM chains 
        (e.g., ConversationalRetrievalChain) recognize it as valid.

        :param search_type: "similarity", "similarity_score_threshold", or "mmr" (not implemented)
        :param search_kwargs: e.g. {"k":4, "score_threshold":0.7}
        """
        if search_kwargs is None:
            search_kwargs = {}
        if "k" not in search_kwargs:
            search_kwargs["k"] = 4

        valid_search_types = ["similarity", "similarity_score_threshold", "mmr"]
        if search_type not in valid_search_types:
            raise ValueError(f"Invalid search_type. Must be one of {valid_search_types}")

        # We use a private attribute for the manager so Pydantic won't try to validate it.
        class CustomVectorStoreRetriever(BaseRetriever):
            search_type: str
            search_kwargs: Dict[str, Any]
            _manager: "VectorStoreManager" = PrivateAttr()

            def __init__(
                self, 
                manager: "VectorStoreManager", 
                search_type: str, 
                search_kwargs: Dict[str, Any],
                **data
            ):
                super().__init__(search_type=search_type, search_kwargs=search_kwargs, **data)
                self._manager = manager

            def get_relevant_documents(self, query: str) -> List[Document]:
                k = self.search_kwargs.get("k", 4)
                results = self._manager.search(query, top_k=k)

                if self.search_type == "similarity":
                    return [
                        Document(page_content=r["document"], metadata=r["metadata"])
                        for r in results
                    ]
                elif self.search_type == "similarity_score_threshold":
                    threshold = self.search_kwargs.get("score_threshold", 0.5)
                    filtered = [r for r in results if r["score"] >= threshold]
                    return [
                        Document(page_content=r["document"], metadata=r["metadata"])
                        for r in filtered
                    ]
                elif self.search_type == "mmr":
                    raise NotImplementedError("MMR not currently supported.")
                else:
                    raise ValueError(f"Unsupported search type: {self.search_type}")

            async def aget_relevant_documents(self, query: str) -> List[Document]:
                # Optional: an async version if the chain calls it
                return self.get_relevant_documents(query)

        return CustomVectorStoreRetriever(
            manager=self,
            search_type=search_type,
            search_kwargs=search_kwargs
        )


    # def as_retriever(
    #     self,
    #     search_type: str = "similarity",
    #     search_kwargs: Optional[dict] = None,
    #     **kwargs
    # ):
    #     """
    #     Returns a VectorStoreRetriever from the underlying vector store, allowing
    #     integration with Retrieval-Augmented Generation (RAG) pipelines.

    #     :param search_type: The type of retrieval/search algorithm to use.
    #            Supported: "similarity", "similarity_score_threshold", "mmr".
    #     :param search_kwargs: Additional search-related configuration (e.g., k=5, score_threshold=0.8, etc.).
    #     :param kwargs: Other optional parameters (e.g., 'tags' for retriever metadata).
    #     :return: A VectorStoreRetriever object that can be used in LLM chains.
    #     """
    #     if search_kwargs is None:
    #         search_kwargs = {}
    #     # The as_retriever(...) method is provided by LangChain's VectorStore base class
    #     return self.vector_store.as_retriever(
    #         search_type=search_type,
    #         search_kwargs=search_kwargs,
    #         **kwargs
    #     )
                    
    def delete_collection(self, collection_name: Optional[str] = None):
        """
        Delete a collection from the vector store.
        :param collection_name: Name of the collection to delete. If None, uses the initialized collection.
        """
        if self.vector_store_type == "chroma":
            if not collection_name:
                collection_name = self.collection_name
            logger.info(f"Deleting Chroma collection '{collection_name}'.")
            chroma_client = self.vector_store.client
            chroma_client.delete_collection(name=collection_name)
            logger.info(f"Chroma collection '{collection_name}' deleted successfully.")
        elif self.vector_store_type == "pinecone":
            logger.info(f"Deleting Pinecone index '{self.pinecone_index_name}'.")
            self.vector_store.delete_index(self.pinecone_index_name)
            logger.info(f"Pinecone index '{self.pinecone_index_name}' deleted successfully.")
        else:
            logger.warning(f"Delete operation not supported for vector_store_type: {self.vector_store_type}")
            
    def update_document(
        self,
        document_id: str,
        updated_document: Document
    ):
        """
        Update a single document in the vector store.
        :param document_id: The unique identifier of the document to update.
        :param updated_document: The updated Document object.
        """
        logger.info(f"Updating document with ID: {document_id}")
        try:
            if self.vector_store_type == "chroma":
                self.vector_store.update_document(document_id=document_id, document=updated_document)
            elif self.vector_store_type == "pinecone":
                embedding = self.embeddings.embed(updated_document.page_content)
                self.vector_store.upsert(vectors=[(document_id, embedding, updated_document.metadata)])
            else:
                raise ValueError(f"Unsupported vector_store_type: {self.vector_store_type}")
            logger.info(f"Document with ID: {document_id} updated successfully.")
        except Exception as e:
            logger.error(f"Failed to update document with ID: {document_id}: {e}")
            
    def update_documents(
        self,
        ids: List[str],
        updated_documents: List[Document]
    ):
        """
        Update multiple documents in the vector store.
        :param ids: List of unique identifiers of the documents to update.
        :param updated_documents: List of updated Document objects.
        """
        logger.info(f"Updating {len(ids)} documents.")
        try:
            if self.vector_store_type == "chroma":
                for doc_id, doc in zip(ids, updated_documents):
                    self.vector_store.update_document(document_id=doc_id, document=doc)
            elif self.vector_store_type == "pinecone":
                vectors = []
                for doc_id, doc in zip(ids, updated_documents):
                    embedding = self.embeddings.embed(doc.page_content)
                    vectors.append((doc_id, embedding, doc.metadata))
                self.vector_store.upsert(vectors=vectors)
            else:
                raise ValueError(f"Unsupported vector_store_type: {self.vector_store_type}")
            logger.info(f"{len(ids)} documents updated successfully.")
        except Exception as e:
            logger.error(f"Failed to update documents: {e}")
            
    def clear_vector_store(self):
        """
        Clear all documents from the vector store.
        """
        logger.info("Clearing all documents from the vector store.")
        try:
            if self.vector_store_type == "chroma":
                self.vector_store.reset()
            elif self.vector_store_type == "pinecone":
                self.vector_store.delete(delete_all=True)
            else:
                raise ValueError(f"Unsupported vector_store_type: {self.vector_store_type}")
            logger.info("Vector store cleared successfully.")
        except Exception as e:
            logger.error(f"Failed to clear vector store: {e}")

    def get_vector_store_info(self) -> Dict:
        """
        Get information about the current vector store.

        :return: Dictionary containing vector store information.
        """
        logger.info("Retrieving vector store information.")
        try:
            if self.vector_store_type == "chroma":
                info = self.vector_store.client.get_collection(collection_name=self.collection_name)
            elif self.vector_store_type == "pinecone":
                info = self.vector_store.describe_index(self.pinecone_index_name)
            else:
                raise ValueError(f"Unsupported vector_store_type: {self.vector_store_type}")
            logger.info("Vector store information retrieved successfully.")
            return info
        except Exception as e:
            logger.error(f"Failed to retrieve vector store information: {e}")
            return {}

    def list_indexes(self) -> List[str]:
        """
        List all available indexes in the vector store.

        :return: List of index names.
        """
        logger.info("Listing all indexes in the vector store.")
        try:
            if self.vector_store_type == "chroma":
                collections = self.vector_store.client.list_collections()
                return collections
            elif self.vector_store_type == "pinecone":
                indexes = self.vector_store.list_indexes()
                return indexes
            else:
                raise ValueError(f"Unsupported vector_store_type: {self.vector_store_type}")
        except Exception as e:
            logger.error(f"Failed to list indexes: {e}")
            return []

    def delete_index(self):
        """
        Delete the current index from the vector store.
        """
        logger.info(f"Deleting index '{self.pinecone_index_name}' from Pinecone.")
        try:
            if self.vector_store_type == "pinecone":
                pinecone.delete_index(self.pinecone_index_name)
                logger.info(f"Pinecone index '{self.pinecone_index_name}' deleted successfully.")
            else:
                logger.warning("Delete index operation is only supported for Pinecone vector store.")
        except Exception as e:
            logger.error(f"Failed to delete Pinecone index '{self.pinecone_index_name}': {e}")

    def get_documents(self, ids: List[str]) -> List[Document]:
        """
        Retrieve documents from the vector store by their IDs.
        """
        logger.info(f"Retrieving {len(ids)} documents from the vector store.")
        try:
            if self.vector_store_type == "chroma":
                documents = self.vector_store.get_documents(ids=ids)
            elif self.vector_store_type == "pinecone":
                results = self.vector_store.fetch(ids=ids)
                documents = [
                    Document(
                        page_content=result.metadata["text"],
                        metadata=result.metadata
                    ) 
                    for result in results
                ]
            else:
                raise ValueError(f"Unsupported vector_store_type: {self.vector_store_type}")
            logger.info(f"Retrieved {len(documents)} documents successfully.")
            return documents
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {e}")
            return []
        

    def get_document_by_id(self, document_id: str) -> Optional[Document]:
        """
        Retrieve a single document from the vector store by its ID.
        """
        logger.info(f"Retrieving document with ID: {document_id}")
        try:
            if self.vector_store_type == "chroma":
                document = self.vector_store.get_document(document_id=document_id)
            elif self.vector_store_type == "pinecone":
                result = self.vector_store.fetch(ids=[document_id])
                if result:
                    document = Document(
                        page_content=result[0].metadata["text"],
                        metadata=result[0].metadata
                    )
                else:
                    document = None
            else:
                raise ValueError(f"Unsupported vector_store_type: {self.vector_store_type}")
            if document:
                logger.info(f"Document with ID: {document_id} retrieved successfully.")
            else:
                logger.warning(f"Document with ID: {document_id} not found.")
            return document
        except Exception as e:
            logger.error(f"Failed to retrieve document with ID: {document_id}: {e}")
            return None


    def delete_documents(self, ids: List[str]):
        """
        Delete documents from the vector store by their IDs.
        """
        logger.info(f"Deleting {len(ids)} documents from the vector store.")
        try:
            if self.vector_store_type == "chroma":
                self.vector_store.delete_documents(ids=ids)
            elif self.vector_store_type == "pinecone":
                self.vector_store.delete(ids=ids)
            else:
                raise ValueError(f"Unsupported vector_store_type: {self.vector_store_type}")
            logger.info(f"{len(ids)} documents deleted successfully.")
        except Exception as e:
            logger.error(f"Failed to delete documents: {e}")


    def delete_document(self, document_id: str):
        """
        Delete a single document from the vector store by its ID.
        """
        logger.info(f"Deleting document with ID: {document_id}")
        try:
            if self.vector_store_type == "chroma":
                self.vector_store.delete_document(document_id=document_id)
            elif self.vector_store_type == "pinecone":
                self.vector_store.delete(ids=[document_id])
            else:
                raise ValueError(f"Unsupported vector_store_type: {self.vector_store_type}")
            logger.info(f"Document with ID: {document_id} deleted successfully.")
        except Exception as e:
            logger.error(f"Failed to delete document with ID: {document_id}: {e}")

