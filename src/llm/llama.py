import os
from dotenv import load_dotenv
from textwrap import dedent
import json
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_ollama import OllamaLLM, ChatOllama, OllamaEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage

load_dotenv()

class LlamaChatbot:
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 512,
        num_threads: int = 4,
        temperature: float = 0.7,
        max_tokens: int = 150,
        top_p: float = 0.9
    ):
        """
        Preserves your original constructor signature, but now we’ll
        instantiate an OllamaLLM for multi-turn usage.
        
        :param model_path:    Path or name of your Llama/Ollama model.
        :param n_ctx:         Context window size.
        :param num_threads:   Threads to use for inference.
        :param temperature:   Sampling temperature.
        :param max_tokens:    Max tokens in the final answer.
        :param top_p:         Nucleus sampling.
        """
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.num_threads = num_threads
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p

        # 1) Create an OllamaLLM instance for the chain
        #    (If you want LlamaCpp instead, replace here.)
        self.llm = OllamaLLM(
            model_path=model_path,
            n_ctx=n_ctx,
            num_threads=num_threads,
            temperature=temperature,
            top_p=top_p,
            # max_tokens param can be passed if supported by your LLM,
            # but not all wrappers support it directly.
        )

        # 2) Prepare memory for multi-turn conversation
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            input_key="question",
            output_key="answer"
        )

        # 3) We'll store our chain once created
        self.conversation_chain = None


    def create_prompt_template(self, template: str, input_variables: list[str]) -> PromptTemplate:
        """
        Returns a basic PromptTemplate.
        """
        return PromptTemplate(template=template, input_variables=input_variables)


    def create_retrieval_qa_chain(self, retriever, chain_type="stuff", prompt_template=None):
        """
        Original method built a RetrievalQA; now we build a ConversationalRetrievalChain
        for RAG. We'll store it in self.conversation_chain.

        :param retriever:       A retriever from your vector store.
        :param chain_type:      Unused, kept for signature compatibility.
        :param prompt_template: (Optional) If provided, a custom ChatPromptTemplate for system + user steps.
        :return:                The conversation chain.
        """
        # If no custom prompt provided, define a simple system prompt referencing {context}
        if prompt_template is None:
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
                HumanMessagePromptTemplate.from_template("{question}"),
            ])
        else:
            # If user provided a PromptTemplate, assume it’s a ChatPromptTemplate or can be used similarly
            chat_prompt = prompt_template

        # Build a multi-turn retrieval chain
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": chat_prompt}
        )
        return self.conversation_chain
    

    def chat(self, messages):
        chat_model = ChatOllama(model=self.model_path, n_ctx=self.n_ctx, num_threads=self.num_threads)
        response = chat_model.invoke(messages)
        return {"result": response}

    def generate_response(self, query, retriever, prompt_template=None, chain_type="stuff"):
        """
        Single-turn usage with conversation chain. We create or update the chain,
        then pass the query as {"question": ...}.

        :param query:           The user's question.
        :param retriever:       VectorStore retriever for doc context.
        :param prompt_template: (Optional) ChatPromptTemplate or similar.
        :param chain_type:      Unused, for signature compatibility.
        :return:                The final answer string.
        """
        self.create_retrieval_qa_chain(retriever, chain_type, prompt_template)
        result = self.conversation_chain({"question": query})
        return result["answer"]

    def generate_response_with_sources(self, query, retriever, prompt_template=None, chain_type="stuff"):
        """
        Same as above, but return the sources too.

        :param query:           The user's question.
        :param retriever:       VectorStore retriever.
        :param prompt_template: Optional ChatPromptTemplate.
        :param chain_type:      Unused, for signature compatibility.
        :return:                Dict with {"result": <answer>, "sources": <list of documents>}
        """
        self.create_retrieval_qa_chain(retriever, chain_type, prompt_template)
        result = self.conversation_chain({"question": query})
        return {
            "result": result["answer"],
            "sources": result["source_documents"]
        }

# Example usage
if __name__ == "__main__":
    # model_path = "./model/zephyr-7b-beta.Q4_0.gguf"
    # Create the LlamaChatbot
    bot = LlamaChatbot(model_path="./models/your-llama-model.bin")

    # Create a prompt template
    template = """
    You are a helpful assistant that answers questions based on the provided context.

    Context:
    {context}

    Question: {query}

    Answer:
    """
    prompt_template = bot.create_prompt_template(template, ["context", "query"])

    # Example conversation
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"}
    ]
    response = chatbot.chat(messages)
    print(response)

    # Example RAG usage
    from langchain.vectorstores import Chroma
    from langchain.embeddings import OllamaEmbeddings

    # Suppose you have a retriever from Pinecone or Chroma
    retriever = my_vectorstore.as_retriever()

    # Single-turn usage
    answer_text = bot.generate_response("What is quantum entanglement?", retriever)
    print("Answer:", answer_text)

    # With sources
    res = bot.generate_response_with_sources("Tell me about black holes", retriever)
    print("Answer:", res["result"])
    print("Sources:", res["sources"])

    # Multi-turn usage
    bot.create_retrieval_qa_chain(retriever)
    messages = [{"role": "user", "content": "Who discovered penicillin?"}]
    resp = bot.chat(messages)
    print("Assistant:", resp)

    # Next user query referencing prior answer
    messages.append({"role": "user", "content": "What year was that?"})
    resp2 = bot.chat(messages)
    print("Assistant:", resp2)
