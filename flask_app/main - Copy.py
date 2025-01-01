from flask import Flask, render_template , jsonify, request
from flask_pymongo import PyMongo
from src.run_rag_pipeline import RunChatbot
import os
from dotenv import load_dotenv
import openai
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

@app.route("/")
def home():
    try:
        chats = mongo.db.chats.find({})  # 'chats' is a collection in the specified database
        myChats = [chat for chat in chats]
        print(myChats)
        return render_template("index.html", myChats=myChats)
    except Exception as e:
        logger.error(f"Error fetching chats: {e}")
        return "An error occurred while fetching chats.", 500

@app.route("/api", methods=["GET", "POST"])
def qa():
    if request.method == "POST":
        try:
            data = request.json
            print(data)
            question = data.get("question")
            if not question:
                return jsonify({"error": "No question provided."}), 400

            chat = mongo.db.chats.find_one({"question": question})
            print(chat)
            if chat:
                response_data = {"question": question, "answer": chat['answer']}
                return jsonify(response_data), 200
            else:
                openai_api_key = os.getenv("OPENAI_API_KEY")
                if not openai_api_key:
                    logger.error("OpenAI API Key is missing.")
                    return jsonify({"error": "OpenAI API Key is missing."}), 400
                openai.api_key = openai_api_key
                response = openai.Completion.create(
                    model="text-davinci-003",
                    prompt=question,
                    temperature=0.7,
                    max_tokens=256,
                    top_p=1,
                    frequency_penalty=0,
                    presence_penalty=0
                )
                print(response)
                answer = response["choices"][0]["text"].strip()
                response_data = {"question": question, "answer": answer}
                mongo.db.chats.insert_one({"question": question, "answer": answer})
                return jsonify(response_data), 200
        except Exception as e:
            logger.error(f"Error in /api route: {e}")
            return jsonify({"error": "An error occurred processing your request."}), 500

    # Handle GET request
    data = {
        "result": "Thank you! I'm just a machine learning model designed to respond to questions and generate text based on my training data. Is there anything specific you'd like to ask or discuss?"
    }
    return jsonify(data), 200

if __name__ == "__main__":
    app.run(debug=True)