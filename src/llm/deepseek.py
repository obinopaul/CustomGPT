import os
from dotenv import load_dotenv
from openai import OpenAI
import openai
from textwrap import dedent
import json
from langchain.chains import RetrievalQA
from langchain_openai import ChatOpenAI as LangchainChatOpenAI
from langchain_openai import OpenAI as LangchainOpenAI
from langchain.prompts import PromptTemplate
from langchain_community.embeddings import OpenAIEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain.schema import SystemMessage, HumanMessage, AIMessage


class DeepSeekChatbot:
    """
    A revised Chatbot class that uses LangChain's ChatOpenAI (no old openai.OpenAI usage).
    """

    def __init__(
        self,
        api_key: str,
        model: str = "deepseek-chat",  # Set a default value
        temperature: float = 0.7,
        max_tokens: int = 150,
        top_p: float = 0.9,
        frequency_penalty: float = 0.3,
        presence_penalty: float = 0.5
    ):
        """
        Initialize the chatbot with OpenAI API key and model configuration.
        
        :param api_key:           OpenAI API key.
        :param model:             OpenAI model name (e.g., "gpt-4").
        :param temperature:       Controls randomness of output.
        :param max_tokens:        Max tokens in the generated response.
        :param top_p:             Nucleus sampling.
        :param frequency_penalty: Penalize repeated tokens.
        :param presence_penalty:  Encourages new topics.
        """
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.api_key = api_key
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.llm = LangchainChatOpenAI(
            api_key=self.api_key if self.api_key else os.getenv("DEEPSEEK_API_KEY"),
            model= self.model if self.model else "deepseek-chat",
            base_url="https://api.deepseek.com",
            temperature=temperature,
            # max_tokens=max_tokens
        )

        # Prepare a memory object for multi-turn conversation
        # (But we won't finalize the chain until we have a retriever)
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",  # must match what ConversationalRetrievalChain expects
            return_messages=True,
            input_key="question",
            output_key="answer"
        )

        # We store the conversation chain here once created
        self.conversation_chain = None
                
    def create_prompt_template(self, template: str, input_variables: list[str]) -> PromptTemplate:
        """
        Create a simple PromptTemplate for custom instructions.

        :param template:        The template string (e.g. "Summarize: {text}")
        :param input_variables: Variables to fill in (e.g. ["text"])
        :return:                A PromptTemplate object usable in chains.
        """
        return PromptTemplate(template=template, input_variables=input_variables)


    def create_prompt_template(self, template: str, input_variables: list[str]) -> PromptTemplate:
        """
        Same signature as original. Returns a simple PromptTemplate.
        """
        return PromptTemplate(template=template, input_variables=input_variables)


    def create_retrieval_qa_chain(self, retriever, chain_type="stuff", prompt_template=None):
        """
        In the original code, this built a RetrievalQA chain. Now we'll build a
        'ConversationalRetrievalChain' using your working pipeline approach.

        :param retriever:       The VectorStore retriever.
        :param chain_type:      (Unused here, but kept for signature compatibility)
        :param prompt_template: Optionally supply a PromptTemplate if desired.
        :return:                The conversation chain (ConversationalRetrievalChain).
        """
        # 1) Define system instructions referencing {context}
        #    We only do this if user didn't provide a custom prompt_template
        if prompt_template is None:
            
            system_template = """
            You are an advanced and knowledgeable AI assistant designed to provide helpful, accurate, and well-explained responses to user queries. You can access and use context provided through retrieval-augmented generation (RAG), but you are also equipped to supplement your responses with general knowledge when context is insufficient, unrelated, or absent.

            {context}
            
            ## Key Directives:
            1. **Context Utilization:**
            - Use the provided context to frame your answers whenever it is directly relevant to the query.
            - If the context contains partial or incomplete information, integrate it appropriately into your response while clearly noting its limitations.
            - If the context does not relate to the user's query, explicitly state: "The provided context is unrelated to this query.", but only state this if you are absolutely sure there is no document that has any relationship to the query, otherwise you are free to use some parts of the document that is related and then supplement your responses so that the user does not lose trust in your provided answers.

            2. **Handling Insufficient or Missing Context:**
            - If no context is provided, or if the context does not address the query:
                - Acknowledge the absence of relevant context. Only if you are absolutely sure there is no document that has any relationship to the query, otherwise you are free to use some parts of the document that is related and then supplement your responses so that the user does not lose trust in your provided answers.
                - Proceed to answer the query using your own general knowledge, ensuring your response is accurate, thorough, and appropriately scoped.

            3. **Balancing Context and General Knowledge:**
            - Clearly differentiate between information derived from the provided context and insights based on your general knowledge. Only if you are absolutely sure there is no document that has any relationship to the query, otherwise you are free to use some parts of the document that is related and then supplement your responses so that the user does not lose trust in your provided answers.
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
            - Response: "Based on general knowledge, the tallest mountain in the world is Mount Everest, which stands at 8,848 meters (29,029 feet)."

            4. **Partial Context:**
            - Query: "Tell me about renewable energy in Europe."
            - Context: "Germany is a leader in solar energy adoption."
            - Response: "Based on the provided context, Germany is a leader in solar energy adoption. Outside of the context provided, Europe also leads in wind and hydroelectric energy, with countries like Denmark and Norway making significant contributions."

            ## Additional Rules:
            - Avoid speculation. If uncertain, say "I am unsure about this, but I can provide general insights or suggest resources."
            - Always strive for completeness while respecting the user's query and context.
            - Ensure your tone remains professional, engaging, and approachable.
            """

            # Build a ChatPromptTemplate with system + user question
            chat_prompt = ChatPromptTemplate.from_messages([
                SystemMessagePromptTemplate.from_template(system_template),
                HumanMessagePromptTemplate.from_template("{question}"),
            ])
        else:
            # If user gave us a custom PromptTemplate, we can wrap it as ChatPromptTemplate if needed
            # For simplicity, let's assume it's already suitable. 
            # If it's not a ChatPromptTemplate, you might adapt it accordingly.
            chat_prompt = prompt_template

        # 2) Build ConversationalRetrievalChain using self.llm + the given retriever
        self.conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": chat_prompt},
        )
        return self.conversation_chain


    # def chat(self, messages, response_format=None):
    #     """
    #     Original signature expects a list of messages dicts. We'll interpret them as
    #     a single user query (the last item) for the conversation chain.

    #     If you truly have a multi-message conversation to pass at once, you could parse them.
    #     But typically, for multi-turn usage, we do repeated calls with "question".

    #     For demonstration, let's just assume 'messages[-1]["content"]' is the question.
    #     """
    #     if not messages:
    #         return "No messages provided."

    #     user_query = messages[-1].get("content", "")
    #     if not user_query:
    #         return "No user query in messages."

    #     if self.conversation_chain is None:
    #         return "Conversation chain not initialized. Please call create_retrieval_qa_chain first."

    #     # Pass the question to the chain
    #     result = self.conversation_chain({"question": user_query})
    #     return result["answer"]
    


    def chat(self, messages, response_format=None):
        """
        Original signature expects a list of messages dicts. We'll interpret them as
        a single user query (the last item) for the conversation chain.
        """
        response = self.llm.invoke(messages)
        return {"result": response}
    
    def generate_response(self, query, retriever, prompt_template=None, chain_type="stuff"):
        """
        Single-turn usage: create (or update) the conversation chain, then get the answer.

        :param query:           The user's question.
        :param retriever:       The retriever with your vector store.
        :param prompt_template: Optional custom ChatPromptTemplate.
        :param chain_type:      (Unused, for signature compatibility).
        :return:                The answer text.
        """
        # Create or update the chain
        self.create_retrieval_qa_chain(retriever, chain_type, prompt_template)

        # Now ask the chain
        result = self.conversation_chain({"question": query})
        return result["answer"]


    def generate_response_with_sources(self, query, retriever, prompt_template=None, chain_type="stuff"):
        """
        Similar to above, but returns sources as well.

        :param query:           The user question.
        :param retriever:       The retriever to fetch docs.
        :param prompt_template: Optional ChatPromptTemplate for chain instructions.
        :param chain_type:      (Unused, for signature compatibility).
        :return:                Dict with {"result": <answer>, "sources": <list of docs>}
        """
        # Create or update the chain
        self.create_retrieval_qa_chain(retriever, chain_type, prompt_template)

        # Ask the chain
        result = self.conversation_chain({"question": query})
        return {
            "result": result["answer"],
            "sources": result["source_documents"]
        }
        

# Example usage
if __name__ == "__main__":
    bot = OpenAIChatbot(api_key=api_key)

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
    response = bot.chat(messages)
    print(response)


    # Assuming you have a Chroma vector store named 'vector_store'
    retriever = vector_store.as_retriever()

    # Single-turn usage
    answer_text = bot.generate_response("What's the capital of France?", retriever=retriever)
    print("Answer:", answer_text)

    # With sources
    res_dict = bot.generate_response_with_sources("Explain quantum entanglement", retriever)
    print("Answer:", res_dict["result"])
    print("Sources:", res_dict["sources"])

    # Multi-turn usage
    bot.create_retrieval_qa_chain(retriever)
    messages = [{"role": "user", "content": "Who discovered penicillin?"}]
    resp = bot.chat(messages)
    print("Assistant:", resp)