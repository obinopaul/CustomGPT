import os
from textwrap import dedent
import json
from dotenv import load_dotenv

# LangChain & HuggingFace
from langchain import hub 
from langchain.chains import ConversationalRetrievalChain
from langchain_huggingface import HuggingFacePipeline
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage

from langchain.prompts import PromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint

load_dotenv()

class HuggingFaceChatbot:
    """
    Revised HuggingFaceChatbot class using a conversational RAG pipeline:
      - HuggingFacePipeline as the LLM
      - ConversationalRetrievalChain w/ {context} in system prompt
      - ConversationBufferMemory for multi-turn conversation
    """

    def __init__(
        self,
        model_name: str,
        huggingface_api_key: str = None,
        device: str = "cpu",
        model_kwargs: dict = None,
        encode_kwargs: dict = None,
        
        **kwargs
    ):
        """
        :param model_name:  The Hugging Face model name or repo id, e.g. "facebook/opt-350m"
        :param api_key:     (Optional) An API key if needed for certain endpoints or private HF repos
        :param device:      Device for inference, e.g. 'cpu' or 'cuda'
        :param model_kwargs:Extra kwargs for the pipeline, e.g. {'max_length': 512}
        :param encode_kwargs:Extra kwargs for embeddings or other HF usage.
        """
        self.model_name = model_name
        self.huggingface_api_key = huggingface_api_key
        self.device = device
        self.model_kwargs = model_kwargs or {}
        self.encode_kwargs = encode_kwargs or {}
        self.kwargs = kwargs

        # 1) Create a HuggingFacePipeline LLM
        #    (If you had a custom endpoint, you'd adapt here.)
        self.llm = HuggingFacePipeline.from_model_id(
            model_id=model_name,
            task="text-generation",
            device=device,
            model_kwargs=self.model_kwargs
            # **self.kwargs
        )

        # Prepare memory for multi-turn chat
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )
        
        # We'll store our ConversationalRetrievalChain here once built
        self.conversation_chain = None
                
                
    def create_prompt_template(self, template: str, input_variables: list[str]) -> PromptTemplate:
        """
        Same signature as before. Returns a basic PromptTemplate.
        """
        return PromptTemplate(template=template, input_variables=input_variables)


    def create_retrieval_qa_chain(self, retriever, chain_type="stuff", prompt_template=None):
        """
        In the old code, this used RetrievalQA. Now we build a multi-turn
        ConversationalRetrievalChain that references {context}.

        :param retriever:       The VectorStore retriever for your docs.
        :param chain_type:      Unused here, kept for method signature compatibility.
        :param prompt_template: If provided, a (Chat)PromptTemplate to override the default system prompt.
        :return:                The created ConversationalRetrievalChain.
        """
        # If user didn't supply a prompt_template, define a system prompt with {context}.
        if not prompt_template:
            
            system_template = """
            You are an advanced and knowledgeable AI assistant designed to provide helpful, accurate, and well-explained responses to user queries. You can access and use context provided through retrieval-augmented generation (RAG), but you are also equipped to supplement your responses with general knowledge when context is insufficient, unrelated, or absent.

            {context}
            
            ## Key Directives:
            1. **Context Utilization:**
            - Use the provided context to frame your answers whenever it is directly relevant to the query.
            - If the context contains partial or incomplete information, integrate it appropriately into your response while clearly noting its limitations.
            - If the context does not relate to the user's query, explicitly state: "The provided context is unrelated to this query."

            2. **Handling Insufficient or Missing Context:**
            - If no context is provided, or if the context does not address the query:
                - Acknowledge the absence of relevant context.
                - Proceed to answer the query using your own general knowledge, ensuring your response is accurate, thorough, and appropriately scoped.

            3. **Balancing Context and General Knowledge:**
            - Clearly differentiate between information derived from the provided context and insights based on your general knowledge.
            - Avoid prioritizing unrelated or conflicting context over general knowledge.

            4. **Transparency and Clarification:**
            - Be explicit about the source of your information:
                - Say "Based on the provided context..." for context-based information.
                - Say "Outside of the context provided..." or "Based on general knowledge..." for other information.
            - Clearly explain the reasoning or logic behind your responses when necessary.

            5. **User-Centric Adaptability:**
            - Tailor responses to the user’s needs and ensure clarity in explanations.
            - When applicable, suggest follow-up actions (e.g., "You might find more information by...") or further clarify uncertainties.

            ## Response Formatting:
            - Start your answer by addressing the query directly.
            - If using context, specify where the relevant information comes from.
            - If context is unrelated or absent, state this and provide an informed response.
            - When applicable, provide step-by-step reasoning or examples to enhance understanding.

            ## Example Behavior:
            1. **Context Available and Relevant:**
            - Query: "What is the capital of France?"
            - Context: "France is a country in Europe with Paris as its capital."
            - Response: "Based on the provided context, the capital of France is Paris."

            2. **Context Available but Irrelevant:**
            - Query: "What is the capital of Germany?"
            - Context: "France is a country in Europe with Paris as its capital."
            - Response: "The provided context is unrelated to the query. Based on general knowledge, the capital of Germany is Berlin."

            3. **No Context Provided:**
            - Query: "What is the tallest mountain in the world?"
            - Context: None.
            - Response: "There is no context provided for this query. Based on general knowledge, the tallest mountain in the world is Mount Everest, which stands at 8,848 meters (29,029 feet)."

            4. **Partial Context:**
            - Query: "Tell me about renewable energy in Europe."
            - Context: "Germany is a leader in solar energy adoption."
            - Response: "Based on the provided context, Germany is a leader in solar energy adoption. Outside of the context provided, Europe also leads in wind and hydroelectric energy, with countries like Denmark and Norway making significant contributions."

            ## Additional Rules:
            - Avoid speculation. If uncertain, say "I am unsure about this, but I can provide general insights or suggest resources."
            - Always strive for completeness while respecting the user's query and context.
            - Ensure your tone remains professional, engaging, and approachable.
            """
            
            chat_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template("{question}")
            ])
        else:
            # If the user gave a custom template, assume it fits the ChatPromptTemplate shape
            chat_prompt = prompt_template

        # Build a ConversationalRetrievalChain for RAG
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": chat_prompt}
        )
        return self.conversation_chain
    

    def chat(self, messages):
        """
        Original code used a HuggingFaceEndpoint call. Now we reuse the conversation chain.

        :param messages: List of message dicts. We'll interpret the last user message as the question.
        :return:         The LLM's response as a string.
        """
        chat_model = HuggingFaceEndpoint(repo_id=self.model_name, api_key=self.huggingface_api_key)
        response = chat_model(messages)
        return {"result": response}

    # def chat(self, messages):
        # """
        # Original code used a HuggingFaceEndpoint call. Now we reuse the conversation chain.

        # :param messages: List of message dicts. We'll interpret the last user message as the question.
        # :return:         The LLM's response as a string.
        # """
    #     if not messages:
    #         return "No messages provided."

    #     user_query = messages[-1].get("content", "")
    #     if not user_query:
    #         return "No user query found in messages."

    #     if not self.conversation_chain:
    #         return "No conversation chain found. Please call create_retrieval_qa_chain first."

    #     result = self.conversation_chain({"question": user_query})
    #     return result["answer"]

    def generate_response(self, query: str, retriever, prompt_template=None, chain_type="stuff"):
        """
        Single-turn usage. We create/update the conversation chain if not already,
        then pass the user query to it.

        :param query:           User's question.
        :param retriever:       VectorStore retriever for doc context.
        :param prompt_template: Optional ChatPromptTemplate.
        :param chain_type:      Ignored, for signature compatibility.
        :return:                The final answer string.
        """
        
        if prompt_template is None:
            prompt = hub.pull("hwchase17/multi-query-retriever")
        # Build or update our conversation chain
        self.create_retrieval_qa_chain(retriever, chain_type, prompt_template)

        # Ask the chain
        result = self.conversation_chain({"question": query})
        return result["answer"]

    def generate_response_with_sources(self, query: str, retriever, prompt_template=None, chain_type="stuff"):
        """
        Return the answer plus source documents.

        :param query:           User's question.
        :param retriever:       VectorStore retriever.
        :param prompt_template: Optional ChatPromptTemplate.
        :param chain_type:      Ignored, for method signature compatibility.
        :return:                {"result": <answer>, "sources": <list of docs>}
        """
        
        if prompt_template is None:
            prompt = hub.pull("hwchase17/multi-query-retriever")
        # Build or update our conversation chain
        self.create_retrieval_qa_chain(retriever, chain_type, prompt_template)

        # Ask the chain
        result = self.conversation_chain({"question": query})
        return {
            "result": result["answer"],
            "sources": result["source_documents"]
        }

# Example usage
if __name__ == "__main__":
    model_name = "gpt2"
    api_key = os.getenv("OPENAI_API_KEY")
    chatbot = HuggingFaceChatbot(model_name, api_key=api_key)

    # Create a prompt template
    template = """
    You are a helpful assistant that answers questions based on the provided context.

    Context:
    {context}

    Question: {query}

    Answer:
    """
    prompt_template = chatbot.create_prompt_template(template, ["context", "query"])

    # Example conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = chatbot.chat(messages)
    print(response)

    # Example RAG usage
    from langchain.vectorstores import Chroma
    from langchain_huggingface import HuggingFaceEmbeddings

    # Assuming you have a Chroma vector store named 'vector_store'
    retriever = vector_store.as_retriever()

    query = "What are the main advantages of using Python for data science?"
    response = chatbot.generate_response(query, retriever, prompt_template)
    print(response)

    response_with_sources = chatbot.generate_response_with_sources(query, retriever, prompt_template)
    print(response_with_sources["result"])
    print("Sources:")
    for source in response_with_sources["sources"]:
        print(source.metadata["source"])