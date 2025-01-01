import streamlit as st
from pymongo import MongoClient
from src.run_rag_pipeline import RunChatbot
import os
import json
import uuid
from dotenv import load_dotenv
import json
import uuid
import logging
import time 
from io import StringIO
import shutil  # Import for folder cleanup

# Load environment variables from .env file
load_dotenv()

# Construct MongoDB URI using environment variables
username = os.getenv("MONGO_USERNAME")
password = os.getenv("MONGO_PASSWORD")
cluster = os.getenv("MONGO_CLUSTER")
database = os.getenv("MONGO_DB")
app_name = os.getenv("MONGO_APP_NAME")

# GitHub API keys
github_username = os.getenv('GITHUB_USERNAME')
github_personal_token = os.getenv('GITHUB_PERSONAL_TOKEN')

# AI models API keys
openai_api_key = os.getenv('OPENAI_API_KEY')
huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY')
huggingface_api_key_2 = os.getenv('HUGGINGFACE_API_KEY_2')

# Vector Database API keys
pinecone_api_key = os.getenv('PINECONE_API_KEY')
pinecone_index_name = os.getenv('PINECONE_INDEX_NAME')

# Research Papers API keys
elsevier_api_key = os.getenv('ELSEVIER_API_KEY')
ieee_api_key =  os.getenv('IEEE_API_KEY')
elsevier_api_secret = os.getenv('ELSEVIER_API_SECRET')

# # MongoDB API keys
# mongo_username = os.getenv('MONGO_USERNAME')
# mongo_password = os.getenv('MONGO_PASSWORD')
# mongo_cluster = os.getenv('MONGO_CLUSTER')
# mongo_db = os.getenv('MONGO_DB')
# mongo_collections = os.getenv('MONGO_COLLECTIONS')
# mongo_app_name = os.getenv('MONGO_APP_NAME')

if not all([username, password, cluster, database, app_name]):
    raise ValueError("One or more MongoDB environment variables are missing.")

# MongoDB Connection URI
mongo_uri = f"mongodb+srv://{username}:{password}@{cluster}/{database}?retryWrites=true&w=majority&appName={app_name}"

# Connect to MongoDB
mongo_client = MongoClient(mongo_uri)

# Access Database and Collections
db = mongo_client[database]
chats_collection = db["chats"]  # Replace "chats" with the name of your collection


# Streamlit App Title
st.title("CustomGPT with RAG Features! 🤖")

# Initialize chatbot
if "chatbot" not in st.session_state:
    st.session_state.chatbot = None
if "messages" not in st.session_state:
    st.session_state.messages = []

# Logger Setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Helper Functions
def initialize_chatbot(
    data_task=None,
    vector_store_type="pinecone",
    model_type="openai",
    data_value=None,
    openai_model=None,
    llama_model_path=None,
    huggingface_model=None,
    embedding_type="openai",
):
    
    if model_type is None:
        raise ValueError("Model type is required but not provided.")
    if model_type == "openai" and openai_model is None:
        raise ValueError("OpenAI model is required but not provided.")
    
    # """Initialize or reinitialize the chatbot instance."""
    st.session_state.chatbot = RunChatbot(
        model_type=model_type,
        api_key=os.getenv("OPENAI_API_KEY"),
        use_rag=bool(data_task),
        data_task=data_task,
        data_value=data_value,
        vector_store_type=vector_store_type,
        embedding_type=embedding_type,
        temperature=0.7,
        github_access_token=os.getenv("GITHUB_PERSONAL_TOKEN"),
        ieee_api_key=os.getenv("IEEE_API_KEY"),
        elsevier_api_key = os.getenv('ELSEVIER_API_KEY'),
        pinecone_api_key=os.getenv("PINECONE_API_KEY"),
        huggingface_api_key = os.getenv('HUGGINGFACE_API_KEY'),
        model=openai_model,
        model_path=llama_model_path,
        model_name=huggingface_model,
    )
    st.session_state.chatbot.setup_data()
    st.session_state.chatbot.setup_vector_store()
    st.session_state.chatbot.setup_llm_pipeline()


def save_uploaded_files(uploaded_files):
    """Save uploaded files and return their paths."""
    if not uploaded_files:
        return []
    session_folder = f"uploads/{uuid.uuid4()}"
    os.makedirs(session_folder, exist_ok=True)
    file_paths = []
    for uploaded_file in uploaded_files:
        file_path = os.path.join(session_folder, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        file_paths.append(file_path)
    return session_folder, file_paths


# Sidebar for RAG Input Options
st.sidebar.header("Additional Input Options")
data_icon = st.sidebar.radio(
    "Select Input Type",
    ["None", "Upload Document", "Web Link", "GitHub Repository", "Research Papers Topic", "Solve GitHub Issues"],
    help="Choose the type of additional input to enhance the chatbot's knowledge base.",
)

uploaded_files = None
data_value = None
data_task = None
rag_ready = False

if data_icon == "Upload Document":
    uploaded_files = st.sidebar.file_uploader("Upload Files", accept_multiple_files=True)
    if uploaded_files:
        data_task = "file"
        session_folder, data_value = save_uploaded_files(uploaded_files)
        st.session_state["session_folder"] = session_folder  # Store the folder path for cleanup
        
elif data_icon == "Web Link":
    web_link = st.sidebar.text_input("Enter Web Link (e.g., 'https://example.com')")
    if web_link.startswith("http://") or web_link.startswith("https://"):
        data_task = "url"
        data_value = web_link
        
elif data_icon == "GitHub Repository":
    github_repo = st.sidebar.text_input("Enter GitHub Repo URL (e.g., 'https://github.com/user/repo.git')")
    if github_repo.startswith("http://") or github_repo.startswith("https://"):
        data_task = "github_repo"
        data_value = github_repo
        
elif data_icon == "Research Papers Topic":
    research_topic = st.sidebar.text_input("Enter Research Paper Topic")
    if research_topic.strip():
        data_task = "research_papers"
        data_value = research_topic
        
elif data_icon == "Solve GitHub Issues":
    github_issues_repo = st.sidebar.text_input("Enter GitHub Repo for Issues (e.g., 'user/repo')")
    if github_issues_repo.strip():
        data_task = "github_issues"
        data_value = github_issues_repo

    
# Sidebar: Machine Learning Models
st.sidebar.header("Large Language Models")
model_category = st.sidebar.radio(
    "Select a Model Category:",
    ["OpenAI", "HuggingFace", "Ollama"],
    help="Choose the category of machine learning model you'd like to use.",
)

openai_model = None
huggingface_model_path = None
ollama_model_path = None
model_ready = False  # Flag to confirm model readiness

# Model Selection
if model_category == "OpenAI":
    openai_model = st.sidebar.selectbox(
        "Choose an OpenAI Model:",
        ["Select a Model", "gpt-4", "gpt-4o", "gpt-4-turbo", "gpt-4o-mini", "o1", "o1-mini", "gpt-3.5-turbo"],
        help="Select an OpenAI model to use."
    )
    if openai_model == "Select a Model":  # Ensure user selects a valid model
        openai_model = None
        st.sidebar.warning("Please select a valid OpenAI model.")
    else:
        st.sidebar.success(f"Selected OpenAI Model: `{openai_model}`")
        model_ready = True

elif model_category == "HuggingFace":
    huggingface_model_path = st.sidebar.text_input(
        "Enter HuggingFace Model Path:",
        placeholder="e.g., google/flan-t5-large",
        help="Provide the path to a HuggingFace model from the model hub."
    )
    if huggingface_model_path.strip():  # Ensure user has provided input
        model_ready = True
        st.sidebar.success(f"Selected HuggingFace Model Path: `{huggingface_model_path}`")
    else:
        huggingface_model_path = None
        st.sidebar.warning("Please provide a valid HuggingFace model path.")

elif model_category == "Ollama":
    ollama_model_path = st.sidebar.text_input(
        "Enter Ollama Model Path:",
        placeholder="e.g., ollama/gpt-j",
        help="Provide the path to an Ollama model."
    )
    if ollama_model_path.strip():  # Ensure user has provided input
        model_ready = True
        st.sidebar.success(f"Selected Ollama Model Path: `{ollama_model_path}`")
    else:
        ollama_model_path = None
        st.sidebar.warning("Please provide a valid Ollama model path.")
        

# Perform RAG or Initialize Model Button
if (data_task and data_value) or model_ready:  # Ensure RAG or Model input is ready
    rag_ready = st.sidebar.button("Perform RAG or Initialize Model")

print(f'Open AI model: {openai_model}')
print(data_task, data_value)

# Initialize chatbot only after confirmation
if rag_ready:  # Check both flags before proceeding
    try:
        if model_category.lower() == "openai" and not openai_model:
            st.error("Please select a valid OpenAI model.")
        elif model_category.lower() == "huggingface" and not huggingface_model_path:
            st.error("Please provide a valid HuggingFace model path.")
        elif model_category.lower() == "ollama" and not ollama_model_path:
            st.error("Please provide a valid Ollama model path.")
        else:
            initialize_chatbot(
                data_task=data_task,
                data_value=data_value,
                model_type=model_category.lower(),
                openai_model=openai_model,
                llama_model_path=ollama_model_path,
                huggingface_model=huggingface_model_path,
            )
            st.sidebar.success(f"Chatbot initialized with model `{model_category}` and RAG task `{data_task}`.")
    finally:
        # Cleanup uploaded files and folder
        session_folder = st.session_state.get("session_folder")
        if session_folder and os.path.exists(session_folder):
            shutil.rmtree(session_folder)  # Remove the folder and its contents
            st.session_state.pop("session_folder", None)  # Clear session state
            st.sidebar.info("Temporary files have been cleaned up.")

# Display Messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat Input
if user_prompt := st.chat_input("Your prompt"):
    st.session_state.messages.append({"role": "user", "content": user_prompt})
    with st.chat_message("user"):
        st.markdown(user_prompt)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        sources_placeholder = st.empty()
        full_response = ""

        try:
            if not st.session_state.chatbot:
                # initialize_chatbot()
                initialize_chatbot(
                    data_task=data_task,
                    data_value=data_value,
                    model_type=model_category.lower(),
                    openai_model=openai_model,
                    llama_model_path=ollama_model_path,
                    huggingface_model=huggingface_model_path,
                )
            result = st.session_state.chatbot.chat(f"Question: {user_prompt}", with_sources=True)

            answer = result.get("result", "Sorry, I couldn't process that.")
            sources = result.get("sources", [])

            if isinstance(answer, str):
                answer = answer
            elif hasattr(answer, 'content'):
                answer = answer.content
                # sources = result.additional_kwargs.get('sources', [])
            else:
                answer = answer['result'].content
                 
        
            # Display the assistant's response
            for token in answer:
                full_response += token
                message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)

            # Display sources if available
            # Display sources if available
            if sources:
                sources_placeholder.markdown("**Sources:**")
                unique_sources = list(set(sources))  # Remove duplicates by converting to a set and back 
                source_links = []
                for idx, source in enumerate(unique_sources, start=1):
                    # Check if the source is a valid URL and display it as a clickable link
                    if isinstance(source, str):
                        if source.startswith("http"):
                            source_links.append(f"{idx}. [{source}]({source})")
                        else:
                            source_links.append(f"{idx}. {source}")
                    else:
                        # Handle non-string sources (e.g., dicts or objects)
                        source_links.append(f"{idx}. {str(source)}")
                # Combine all source links and display them
                sources_placeholder.markdown("\n".join(source_links))

                            
            # Save the response to session and MongoDB
            st.session_state.messages.append({"role": "assistant", "content": full_response})
            chats_collection.insert_one(
                {
                    "question": user_prompt,
                    "answer": full_response,
                    "sources": sources,
                    "data_task": data_task,
                    "data_value": data_value,
                }
            )
        except Exception as e:
            st.error(f"An error occurred: {e}")
            logger.error(f"Error: {e}")

# Clear Chat History
if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    chats_collection.delete_many({})
    st.sidebar.success("Chat history cleared.")
