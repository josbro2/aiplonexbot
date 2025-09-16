from flask import Flask, request, jsonify
from dotenv import load_dotenv
import os
from langchain_community.chat_models import ChatGoogleGenerativeAI

from langchain.prompts.chat import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferMemory

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)

# Store session memories in memory
session_store = {}

# Initialize Gemini model
llm = ChatGoogleGenerativeAI(
    model="models/gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.7
)

# Prompt template
prompt = ChatPromptTemplate.from_messages([
    ("system", """
You are an AI Agent for Aiplonex. You speak in Marathi and English naturally.
You offer:
- AI Agent Development
- AI Chatbot Development
- Website Development
- App Development
- UI/UX Design

Your task is to:
- Be polite, friendly, and professional.
- Understand if the user is a potential client.
- Explain benefits simply.
- Offer to schedule a free consultation.
"""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

# Helper function to get or create memory per session
def get_memory(session_id: str) -> ConversationBufferMemory:
    if session_id not in session_store:
        session_store[session_id] = ConversationBufferMemory(memory_key="history", return_messages=True)
    return session_store[session_id]

# Root route for testing
@app.route("/", methods=["GET"])
def home():
    return jsonify({"status": "Aiplonex Gemini AI Agent is running"})

# Chat route
@app.route("/chat", methods=["GET", "POST"])
def chat():
    if request.method == "GET":
        return jsonify({"status": "Send a POST request with {session_id, message}"})

    data = request.get_json()
    session_id = data.get("session_id", "default_user")  # Default session ID
    user_input = data.get("message", "").strip()

    if not user_input:
        return jsonify({"error": "Message is required"}), 400

    # Get session memory
    memory = get_memory(session_id)

    # Create the chain
    chain = LLMChain(
        llm=llm,
        prompt=prompt,
        memory=memory
    )

    try:
        # Run the chain with user input
        reply = chain.run(user_input)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
