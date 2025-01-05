# llm_pipeline.py
from typing import Dict, List, Optional
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from src.llm.openai import OpenAIChatbot
from src.llm.deepseek import DeepSeekChatbot
from src.llm.huggingface import HuggingFaceChatbot
from src.llm.llama import LlamaChatbot
import os

class LLMPipeline:
    def __init__(
        self,
        model_type: str,
        api_key: Optional[str] = None,
        model_path: Optional[str] = None,
        model_name: Optional[str] = None,
        memory_type: str = "buffer",         # (No longer used here, but kept for signature)
        window_size: int = 5,
        return_messages: bool = True,
        k: int = 4,
        huggingface_api_key = None,
        deepseek_api_key = None,
        deepseek_model = None,
        model: Optional[str] = None,
        **kwargs
    ):
        """
        A pipeline that chooses one of three chatbot classes (OpenAI, HuggingFace, Llama)
        and delegates conversation logic to it.

        :param model_type:      "openai" | "huggingface" | "llama"
        :param api_key:         Required for OpenAI or HuggingFace private repos
        :param model_path:      For local Llama or huggingface model
        :param model_name:      For huggingface repos or custom naming
        :param memory_type:     (Deprecated here, as memory is inside each chatbot)
        :param window_size:     (Deprecated, similarly)
        :param return_messages: (Deprecated)
        :param k:               (Deprecated)
        :param kwargs:          Additional config passed to chatbot constructor
        """
        self.model_type = model_type
        self.model = model
        self.api_key = api_key
        self.model_path = model_path
        self.model_name = model_name
        self.kwargs = kwargs
        self.deepseek_model = deepseek_model
        self.huggingface_api_key = huggingface_api_key
        self.deepseek_api_key = deepseek_api_key

        # Instantiate the appropriate chatbot
        self.chatbot = self._load_chatbot()

    def _load_chatbot(self):
        """
        Returns an instance of OpenAIChatbot, HuggingFaceChatbot, or LlamaChatbot
        based on model_type.
        """
        if self.model_type == "openai":
            return OpenAIChatbot(api_key=self.api_key, 
                                 model=self.model,
                                 **self.kwargs)
        elif self.model_type == "deepseek":
            return DeepSeekChatbot(api_key=self.deepseek_api_key if self.deepseek_api_key else os.getenv("DEEPSEEK_API_KEY",), 
                                 model = self.deepseek_model,
                                 **self.kwargs)
        elif self.model_type == "huggingface":
            return HuggingFaceChatbot(
                model_name=self.model_name,
                huggingface_api_key=self.huggingface_api_key,
                **self.kwargs
            )
        elif self.model_type == "ollama":
            return LlamaChatbot(
                model_path=self.model_path,
                **self.kwargs
            )
        else:
            raise ValueError(f"Unsupported model type: {self.model_type}")


    def run(self, query: str) -> Dict[str, str]:
        """
        Runs a single-turn conversation using the chatbot's 'chat' method,
        passing the query as the user's message.
        """
        # We mimic the old usage that returned {"query", "response"}.
        # In your original code, you used self.conversation_chain.run(query).
        # Now, delegate to the chatbot’s 'chat' method with a single user message.
        messages = [{"role": "user", "content": query}]
        response_text = self.chatbot.chat(messages)
        return {"query": query, "response": response_text}


    def chat(self, messages: List[Dict[str, str]]) -> Dict[str, str]:
        """
        For multi-turn conversation. 
        The input is a list of { "user": "...", "assistant": "..." } or similar.
        We'll convert it to the chatbot's expected format: 
            [ {"role": "user"/"assistant", "content": "..."} ]
        Then call self.chatbot.chat(...)
        """
        # Convert your format to a list of role-based messages
        # Original code: {"user": "..."} = user msg, {"assistant": "..."} = assistant msg
        role_messages = []
        for msg in messages:
            # If you have keys "user"/"assistant", we map them to "role"
            if "user" in msg:
                role_messages.append({"role": "user", "content": msg["user"]})
            if "assistant" in msg:
                role_messages.append({"role": "assistant", "content": msg["assistant"]})

        # We only need to respond to the last user message
        response_text = self.chatbot.chat(role_messages)
        return {
            "query": messages[-1].get("user", ""),
            "response": response_text
        }

    def generate_response(self, query: str, retriever: Chroma) -> Dict[str, str]:
        """
        Delegates to chatbot.generate_response(...),
        which uses a RAG pipeline internally.
        """
        response_text = self.chatbot.generate_response(query, retriever)
        return {"query": query, "response": response_text}

    def generate_response_with_sources(self, query: str, retriever: Chroma) -> Dict[str, str]:
        """
        Delegates to chatbot.generate_response_with_sources(...),
        returning answer + sources.
        """
        resp = self.chatbot.generate_response_with_sources(query, retriever)
        return resp


# Example usage
if __name__ == "__main__":
    # OpenAI pipeline
    openai_pipeline = LLMPipeline(model_type="openai", api_key="your_openai_api_key", temperature=0.7)
    openai_response = openai_pipeline.run("What's the capital of France?")
    print("OpenAI Response:")
    print(openai_response)

    # Hugging Face pipeline
    huggingface_pipeline = LLMPipeline(model_type="huggingface", model_name="gpt2", api_key="your_huggingface_api_key", device="cpu")
    huggingface_response = huggingface_pipeline.run("What's the currency of Japan?")
    print("Hugging Face Response:")
    print(huggingface_response)

    # Llama pipeline
    llama_pipeline = LLMPipeline(model_type="ollama", model_path="./model/llama-7b", n_ctx=512, num_threads=4)
    llama_response = llama_pipeline.run("What's the largest continent in the world?")
    print("Llama Response:")
    print(llama_response)

    # Chat example
    messages = [
        {"user": "Hi, how are you?", "assistant": "I'm doing well, thank you! How can I assist you today?"},
        {"user": "Can you tell me a joke?", "assistant": "Sure! Here's a joke for you: Why don't scientists trust atoms? Because they make up everything!"},
        {"user": "That's a good one! Do you know any other science jokes?", "assistant": ""}
    ]
    chat_response = openai_pipeline.chat(messages)
    print("Chat Response:")
    print(chat_response)

    # Retrieval-augmented generation example

    # Assuming you have a Chroma vector store named 'vector_store'
    retriever = vector_store.as_retriever()

    query = "What are the main advantages of using Python for data science?"
    rag_response = openai_pipeline.generate_response(query, retriever)
    print("Retrieval-Augmented Generation Response:")
    print(rag_response)

    rag_response_with_sources = openai_pipeline.generate_response_with_sources(query, retriever)
    print("Retrieval-Augmented Generation Response with Sources:")
    print(rag_response_with_sources["result"])
    print("Sources:")
    for source in rag_response_with_sources["sources"]:
        print(source.metadata["source"])