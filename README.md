# CustomGPT Chatbot with RAG Features

## Overview
This project is a **custom chatbot application** built with **Streamlit**, integrating advanced machine learning capabilities using **OpenAI**, **HuggingFace**, and **Ollama** models. The chatbot supports various **Retrieval-Augmented Generation (RAG)** operations to enhance its responses and is powered by a flexible backend that uses **Pinecone** as the default vector database (with support for Chroma as well). Chat history is stored in **MongoDB**, enabling future features like returning common prompts and responses.

This project was developed during the Christmas holidays of 2024 and marks my final project for the year. 

---

<div align="center">
  <img src="CustomGPT.gif" alt="Alt Text" />
</div>


## Key Features

### 1. **Model Support**
The chatbot can be configured to work with the following model categories:
- **OpenAI:** Supports models like `gpt-4`, `gpt-4-turbo`, and `gpt-3.5-turbo`.
- **HuggingFace:** Allows users to input a model path from the HuggingFace Model Hub (e.g., `google/flan-t5-large`).
- **Ollama:** Supports Ollama models with user-provided paths.

### 2. **Retrieval-Augmented Generation (RAG)**
The chatbot supports the following RAG operations to enhance its knowledge base and responses:
- **Upload Document:** Upload documents to extract relevant context for responses.
- **Web Link:** Input a URL to fetch information and build context.
- **GitHub Repository:** Fetch details from a specified GitHub repository for relevant queries.
- **Solve GitHub Issues:** Fetch details from issues and solutions from a given GitHub repository and use it for relevant queries.
- **Research Papers Topic:** Search for and retrieve insights from research papers across platforms like Arxiv, Elsevier, and IEEE.


### 3. **Vector Database Integration**
- **Pinecone** (default): Provides fast and scalable vector similarity search.
- **Chroma:** A lightweight alternative for local vector search.

The flexibility to choose between these databases makes the chatbot adaptable for different use cases and infrastructures.

### 4. **Chat History Storage**
The chatbot stores all conversation histories in **MongoDB**, enabling users to:
- Track conversations.
- Potentially implement features like suggesting frequently asked prompts.

### 5. **Streamlit Frontend**
The chatbot features a simple and intuitive **Streamlit-based user interface**, allowing users to:
- Select models and configure their settings dynamically.
- Perform RAG operations easily.
- Chat with the bot and view responses and their sources.

---

## Prerequisites

1. **Install Dependencies**  
   Install the required Python libraries from `requirements.txt`:
   ```bash
   pip install -r requirements.txt
   ```

2. **API Keys**  
   To use the chatbot, you need the following API keys:
   * **OpenAI API Key**
   * **HuggingFace API Key**
   * **Pinecone API Key**
   * **GitHub Personal Access Token**
   * **Elsevier and IEEE API Key** (for research papers retrieval, it defaults to Arxive if no Research API is provided)

   Use the `.env.example` file as a template to create your `.env` file. Replace the placeholders with your actual API keys.

3. **MongoDB**  
   Set up a MongoDB database to store chat histories. Add your MongoDB credentials to the `.env` file.

## Getting Started

1. **Clone the Repository**
   ```bash
   git clone https://github.com/obinopaul/CustomGPT.git
   cd CustomGPT
   ```

2. **Set Up the Environment**  
   Create a `.env` file using the provided `.env.example` and add your API keys and MongoDB credentials.

3. **Run the Application**  
   Start the Streamlit app:
   ```bash
   streamlit run app.py
   ```

4. **Interact with the Chatbot**
   * Select a model category (**OpenAI**, **HuggingFace**, or **Ollama**) and configure the model
   * Perform RAG operations to enhance the chatbot's knowledge base
   * Type queries in the chat interface and get detailed responses with sources



## Conclusion

This custom chatbot is a versatile tool, combining state-of-the-art language models with dynamic RAG capabilities. It was a rewarding project to close out 2024, built during the Christmas holidays as a demonstration of advanced AI integration."# CustomGPT" 
