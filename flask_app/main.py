from flask import Flask, render_template , jsonify, request
from flask_pymongo import PyMongo
from src.run_rag_pipeline import RunChatbot
import os
from dotenv import load_dotenv
import openai
from werkzeug.utils import secure_filename
import json
import uuid
import logging

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)  

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

# MongoDB API keys
mongo_username = os.getenv('MONGO_USERNAME')
mongo_password = os.getenv('MONGO_PASSWORD')
mongo_cluster = os.getenv('MONGO_CLUSTER')
mongo_db = os.getenv('MONGO_DB')
mongo_collections = os.getenv('MONGO_COLLECTIONS')
mongo_app_name = os.getenv('MONGO_APP_NAME')

if not all([username, password, cluster, database, app_name]):
    logger.error("One or more MongoDB environment variables are missing.")
    raise ValueError("One or more MongoDB environment variables are missing.")

app.config["MONGO_URI"] = f"mongodb+srv://{username}:{password}@{cluster}/{database}?retryWrites=true&w=majority&appName={app_name}" 

mongo = PyMongo(app)

# Uploads directory
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize chatbot instance
global_chatbot = None

def initialize_chatbot(data_task=None, data_value=None):
    global global_chatbot
    global_chatbot = RunChatbot(
        model_type="openai",    # or "openai", "llama", "huggingface"
        api_key=openai_api_key,
        use_rag=bool(data_task),
        data_task=data_task,  # or "url", "github_repo", etc.
        data_value=data_value,  # path to the folder  "./data for file"
        vector_store_type="pinecone",
        embedding_type="openai",
        temperature=0.7,
        github_access_token=github_personal_token,
        ieee_api_key=ieee_api_key,
        pinecone_api_key=pinecone_api_key,
    )
    global_chatbot.setup_data()
    global_chatbot.setup_vector_store()
    global_chatbot.setup_llm_pipeline()


@app.route("/")
def home():
    try:
        chats = mongo.db.chats.find({})  # 'chats' is a collection in the specified database
        myChats = [chat for chat in chats]
        return render_template("index.html", myChats=myChats)
    except Exception as e:
        logger.error(f"Error fetching chats: {e}")
        return "An error occurred while fetching chats.", 500


def save_uploaded_files(uploaded_files):
    """Save uploaded files to a unique session folder and return their paths."""
    if not uploaded_files:
        return []

    uploaded_file_paths = []
    session_folder = os.path.join(UPLOAD_FOLDER, str(uuid.uuid4()))
    os.makedirs(session_folder, exist_ok=True)

    for uploaded_file in uploaded_files:
        filename = secure_filename(uploaded_file.filename)
        if filename:
            upload_path = os.path.join(session_folder, filename)
            uploaded_file.save(upload_path)
            uploaded_file_paths.append(upload_path)

    # logger.info(f"Files saved to: {uploaded_file_paths}")
    return uploaded_file_paths


@app.route("/api/upload", methods=["POST"])
def upload_files():
    try:
        uploaded_files = request.files.getlist("file")

        if not uploaded_files:
            return jsonify({"error": "No files provided."}), 400

        uploaded_file_paths = save_uploaded_files(uploaded_files)
        logger.info(f"Files saved to: {uploaded_file_paths}")
        
        if not uploaded_file_paths:
            return jsonify({"error": "Failed to save files."}), 500

        return jsonify({"message": "Files uploaded successfully.", "file_paths": uploaded_file_paths}), 200

    except Exception as e:
        logger.error(f"Error uploading files: {e}")
        return jsonify({"error": "An error occurred during file upload."}), 500




@app.route("/clear_chats", methods=["POST"])
def clear_chats():
    try:
        mongo.db.chats.delete_many({})  # Deletes all documents in the 'chats' collection
        return jsonify({"message": "Chat history cleared successfully."}), 200
    except Exception as e:
        logger.error(f"Error clearing chat history: {e}")
        return jsonify({"error": "An error occurred while clearing chat history."}), 500


@app.route("/api", methods=["GET", "POST"])
def qa():
    global global_chatbot

    try:
        # Retrieve data from the request
        question = request.form.get("question")
        uploaded_file_paths = request.form.get("file_paths")  # File paths from the upload response
        # upload_files = request.form.get("file_path")
        web_link = request.form.get("Web Link")
        clone_github_repo = request.form.get("Clone GitHub Repo")
        research_papers_topic = request.form.get("Research Papers Topic")
        solve_github_issues = request.form.get("Solve GitHub Issues")

        # Deserialize file_paths from JSON
        if uploaded_file_paths:
            uploaded_file_paths = json.loads(uploaded_file_paths)

        # Determine additional input type
        data_task = None
        data_value = None
        
        if uploaded_file_paths:
            data_task = "file"
            data_value = uploaded_file_paths
            # uploaded_file_paths = save_uploaded_files(uploaded_files)  # Save files and get their paths
            logger.info(f"Files saved to: {uploaded_file_paths}")
        elif web_link:
            data_task = "url"
            data_value = web_link
        elif research_papers_topic:
            data_task = "research_papers"
            data_value = research_papers_topic
        elif clone_github_repo:
            data_task = "github_repo"
            data_value = clone_github_repo
        elif solve_github_issues:
            data_task = "github_issues"
            data_value = solve_github_issues
        else:
            data_task = None
            data_value = None
            
        # Log received inputs
        logger.info(f"Question: {question}")
        # logger.info(f"Upload Document: {upload_document.filename if upload_document else 'None'}")
        logger.info(f"Uploaded Files: {uploaded_file_paths}")
        logger.info(f"Web Link: {web_link}")
        logger.info(f"Clone GitHub Repo: {clone_github_repo}")
        logger.info(f"Research Papers Topic: {research_papers_topic}")
        logger.info(f"Solve GitHub Issues: {solve_github_issues}")

        print("data_task: ", data_task)
        print("data_value: ", data_value)
        
        
        # Ensure a question is provided
        if not question:
            return jsonify({"error": "No question provided."}), 400

        # Ensure data_task and data_value are valid
        if data_task == "file" and not os.path.exists(data_value):
            return jsonify({"error": f"File {data_value} does not exist."}), 400
        
        # Initialize chatbot if not already initialized or if additional input is provided
        if not global_chatbot or data_task:
            initialize_chatbot(data_task=data_task, data_value=data_value)

        # Check if the question exists in MongoDB
        chat = mongo.db.chats.find_one({"question": question})
        if chat:
            return jsonify({"question": question, "answer": chat["answer"]}), 200

        # Generate an answer using the chatbot instance
        prompt = f"Question: {question}\n"
        result = global_chatbot.chat(prompt, with_sources=True)

        answer = result['result']

        if isinstance(answer, str):
            answer = answer
        elif hasattr(answer, 'content'):
            answer = answer.content
            sources = result.additional_kwargs.get('sources', [])
        else:
            answer = answer['result'].content
            
        sources = result.get('sources', "") 

        # Save the question and answer to MongoDB
        mongo.db.chats.insert_one({
            "question": question,
            "answer": answer,
            "sources": sources,
            "additional_inputs": {
                "web_link": web_link,
                "clone_github_repo": clone_github_repo,
                "research_papers_topic": research_papers_topic,
                "solve_github_issues": solve_github_issues
            }
        })
        
        # Delete uploaded files after processing
        if uploaded_file_paths:
            for file_path in data_value:
                os.remove(file_path)
            os.rmdir(UPLOAD_FOLDER) # Remove the session folder

        return jsonify({"question": question, "answer": answer, "sources": sources}), 200

    except Exception as e:
        logger.error(f"Error in /api route: {e}")
        return jsonify({"error": "An error occurred processing your request."}), 500

if __name__ == "__main__":
    initialize_chatbot()  # Initialize chatbot once at the start
    app.run(debug=True)
