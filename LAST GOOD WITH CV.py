from pydoc import text
import string
from flask import Flask, request, jsonify
import pandas as pd
import random
import os
from langchain.memory import ConversationBufferMemory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_mistralai.chat_models import ChatMistralAI
from dotenv import load_dotenv
import PyPDF2
from langchain.schema import HumanMessage, AIMessage  # Import these classes
import requests
from flask_socketio import SocketIO
import logging
import threading
from queue import Queue
from termcolor import colored
import time
from feature_extractor import extract_features_from_video
from send_message_to_ADK import send_message_to_adk #New function for sending messages to ADK

load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app)

# Redirect Flask logs and custom logs to the same file
log = logging.getLogger('werkzeug')
file_handler = logging.FileHandler('flask_logs.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

# Set up a custom logger for candidate and interviewer messages
app_logger = logging.getLogger('app_logger')
app_logger.setLevel(logging.INFO)
app_logger.addHandler(file_handler)

# Function to extract text from PDF CV
def generate_id(length=16):
    characters = string.ascii_letters + string.digits  # A-Z, a-z, 0-9
    id = ''.join(random.choices(characters, k=length))
    return id

def extract_cv_text(pdf_path):
    try:
        with open(pdf_path, 'rb') as file:
            reader = PyPDF2.PdfReader(file)
            text = ""
            for page in reader.pages:
                text += page.extract_text()
            return text
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""

# Load technical questions from CSV
def load_technical_questions(csv_path):
    df = pd.read_csv(csv_path, encoding="latin1")
    return df[['Question', 'Answer']].to_dict('records')

# Initialize LLMs
greeting_llm = ChatMistralAI(model_name="mistral-large-latest", temperature=0.7)
behavioral_llm = ChatMistralAI(model_name="mistral-large-latest", temperature=0.6)
technical_llm = ChatMistralAI(model_name="mistral-large-latest", temperature=0.5)

# Create prompt templates
def get_greeting_prompt(cv_text):
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are a professional interviewer conducting a job interview. 
        The candidate's CV contains this information:
        {cv_text}
        Maintain a friendly but professional tone."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
behavioral_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are conducting a behavioral interview. Ask questions that reveal 
    the candidate's soft skills, problem-solving approach, and cultural fit.
    Ask follow-up questions when interesting points emerge."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

technical_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a technical interviewer. Ask questions precisely and 
    evaluate answers professionally."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


# Extract CV text and load questions    
cv_path = os.path.join("Interview Files", "candidate_cv.pdf")  # Adjusted to the Interview Files directory
csv_path = os.path.join("Interview Files", "Software Questions.csv")  # Adjusted to the Interview Files directory
cv_text = extract_cv_text(cv_path)

greeting_chain = get_greeting_prompt(cv_text) | greeting_llm
behavioral_chain = behavioral_prompt | behavioral_llm
technical_chain = technical_prompt | technical_llm

# Load technical and behavioral questions
tech_questions = load_technical_questions(csv_path)
behavioral_questions = [
        "Tell me about a time you faced a difficult challenge at work and how you handled it",
        "Describe a situation where you had to work with a difficult team member",
        "Give an example of when you took initiative to improve a process"
    ]

def serialize_messages(messages):
    serialized = []
    for message in messages:
        if isinstance(message, HumanMessage):
            serialized.append({"role": "user", "content": message.content})
        elif isinstance(message, AIMessage):
            serialized.append({"role": "ai", "content": message.content})
        else:
            serialized.append({"role": "unknown", "content": str(message)})
    return serialized

# Global memory to store conversation history
conversation_memory = ConversationBufferMemory(return_messages=True)

class InterviewUser:
    def __init__(self, socket_id, user_id, name):
        self.socket_id = socket_id
        self.user_id = user_id
        self.session_id = generate_id()  # Generate a unique session ID
        self.name = name
        self.adk_connected= False
        self.phase = "greeting"
        self.question_count=0

    def __repr__(self):
        return f"InterviewUser(socket_id={self.socket_id})"


def create_app():
    app = Flask(__name__)
    socketio.init_app(app)
    return app
users = []
def get_user_by_socket_id(socket_id):
    for user in users:
        if user.socket_id == socket_id:
            return user
    return None

@socketio.on('connect')
def handle_connect():
    print("Node.js socket connected")

@socketio.on('disconnect')
def handle_disconnect():
    print('A client disconnected!')

def invoke_with_rate_limit(chain, input_data):
    time.sleep(1)  # Add a 1-second delay between requests
    return chain.invoke(input_data)

@socketio.on('start_interview')
def handle_start_interview(data):
    print(colored("-----------START INTERVIEW-----------", 'cyan'))
    print(data)
    user=InterviewUser(data['socketId'], data['userId'], data['name'])
    users.append(user) 
    try:
        response = requests.post(url=f"http://localhost:8000/apps/CODEEVAL/users/{data['userId']}/sessions/{user.session_id}",json={})
        if response.status_code == 200:
            user.adk_connected = True
            print(f"ADK connected for user {data['userId']} with session ID {user.session_id}")
        else:
            print(f"User {data['userId']} failed to connect to ADK. Status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        socketio.emit('ai_response', {"phase": "error", "response": "A network error occurred. Please try again later.", "recipient": data['socketId']})
        return
    response = invoke_with_rate_limit(greeting_chain, {
        "input": "Greet the candidate warmly.",
        "history": []
    })
    conversation_memory.chat_memory.add_user_message("Greet the candidate warmly")
    conversation_memory.chat_memory.add_ai_message(str(response.content))
    
    # Log the AI's response to the console and to the log file
    print(f"{colored('Interviewer:', 'cyan')} {response.content}")
    app_logger.info(f"Interviewer: {response.content}")
    
    # Always use the original socketId from the event data
    socketio.emit('ai_response', {"phase": "greeting", "response": response.content, "recipient": data['socketId']})

def wait_for_candidate_response(socket_id):
    """Waits for a candidate's response from the socket."""
    response_queue = Queue()

    def on_response(data):
        if data['socketId'] == socket_id:
            response_queue.put(data['message'])

    # Set up a temporary event listener
    socketio.on('candidate_response', on_response)

    # Block until a response is received
    response = response_queue.get()

    # Remove the temporary listener
    socketio.off('candidate_response', on_response)

    return response
    
def small_talk(user,user_input):
   conversation_memory.chat_memory.add_user_message(user_input)
        
   history = conversation_memory.chat_memory.messages
   response = invoke_with_rate_limit(greeting_chain, {
    "input": user_input,
    "history": history
  })
   print(f"{response.content}")
   conversation_memory.chat_memory.add_ai_message(response.content) 
   socketio.emit('ai_response', { "phase": user.phase, "response": response.content, "recipient": user.socket_id})

def ask_behavioural(user, user_input):
    # Use user_input to influence the first question or response
    history = conversation_memory.chat_memory.messages
    
    # Generate the initial response based on user_input

    # Proceed with the selected behavioral questions
    selected_behavioral = random.sample(behavioral_questions, min(3, len(behavioral_questions)))

    for question in selected_behavioral:
        history = conversation_memory.chat_memory.messages
        
        # Generate and send each behavioral question
        response = invoke_with_rate_limit(behavioral_chain, {
            "input": f"""Comment briefly on the last thing they said and then ask: {question}
            Don't mention the interview process or thank them.""",
            "history": history
        })
        print(f"{colored('Interviewer:', 'cyan')} {response.content}")


        conversation_memory.chat_memory.add_ai_message(response.content)
        socketio.emit('ai_response', { 
            "phase": user.phase, 
            "response": response.content, 
            "recipient": user.socket_id 
        })

        # Wait for the candidate's answer via socket
        candidate_answer = wait_for_candidate_response(user.socket_id)
        conversation_memory.chat_memory.add_user_message(candidate_answer)

        # Generate a follow-up based on the candidate's answer
        history = conversation_memory.chat_memory.messages
        follow_up = invoke_with_rate_limit(behavioral_chain, {
            "input": f"Generate one relevant follow-up based on: {candidate_answer}",
            "history": history
        })

        # Send the follow-up to the candidate
        conversation_memory.chat_memory.add_ai_message(follow_up.content)
        socketio.emit('ai_response', { 
            "phase": user.phase, 
            "response": follow_up.content, 
            "recipient": user.socket_id 
        })

        # Wait for the candidate's follow-up answer via socket
        candidate_followup = wait_for_candidate_response(user.socket_id)
        conversation_memory.chat_memory.add_user_message(candidate_followup)


def ask_technical(user, user_input):
    selected_technical = random.sample(tech_questions, 3)

    # Use user_input to influence the first question or response
    history = conversation_memory.chat_memory.messages

    # Proceed with remaining technical questions
    for question_data in selected_technical[1:]:
        question = question_data['Question']
        model_answer = question_data['Answer']

        # Generate the technical question
        response = invoke_with_rate_limit(technical_chain, {
            "input": f"Ask this technical question: {question}",
            "history": history
        })
        print(f"{colored('Interviewer:','cyan')} {response.content}")

        # Send the question to the candidate
        conversation_memory.chat_memory.add_ai_message(response.content)
        socketio.emit('ai_response', {
            "phase": user.phase,
            "response": response.content,
            "recipient": user.socket_id
        })

        # Wait for candidate's answer
        candidate_answer = wait_for_candidate_response(user.socket_id)
        conversation_memory.chat_memory.add_user_message(candidate_answer)

        # Evaluate the candidate's answer
        evaluation = invoke_with_rate_limit(technical_chain, {
            "input": f"""Evaluate this answer for '{question}':
            Expected: {model_answer}
            Candidate: {candidate_answer}
            Provide specific feedback and score from 1-10""",
            "history": history
        })

        # Send the evaluation to the candidate
        socketio.emit('evaluation', {
            "phase": user.phase,
            "response": evaluation.content,
            "recipient": user.socket_id
        })

def phase_transition(user,user_input):
    history = conversation_memory.chat_memory.messages
    prompt ={
            "input": f"The last point in your {user.phase} phase of the interview, the user said: \"{user_input}\", Comment on it briefly and do not ask any questions.",
            "history": history
        }
    match user.phase:
        case "greeting":
            response = invoke_with_rate_limit(greeting_chain, prompt)
        case "behavioural":
            response = invoke_with_rate_limit(behavioral_chain, prompt)
        case "technical":
            response = invoke_with_rate_limit(technical_chain, prompt)

    print(f"{colored('Interviewer:','cyan')} {response.content}")
    conversation_memory.chat_memory.add_ai_message(response.content) 
    socketio.emit('ai_response', { "phase": user.phase, "response": response.content, "recipient": user.socket_id})
    
    match user.phase:
        case "greeting":
            user.phase = "behavioural"
            print(colored("-----------BEHAVIOURAL PHASE-----------",'cyan'))
            ask_behavioural(user, user_input)
        case "behavioural":
            user.phase = "technical"
            print(colored("-----------TECHNICAL PHASE-----------",'cyan'))
            ask_technical(user, user_input)
        case "technical":
            user.phase = "end"

@socketio.on('message')
def handle_message(data):
    user = get_user_by_socket_id(data['socketId'])
    print(colored(f"{user.name}: ","yellow") + data['message'])
    app_logger.info(f"{user.name}: {data['message']}")
    user.question_count += 1
    if user.phase == "greeting":
        
        if user.question_count < 3:
            small_talk(user, data['message'])
        else:
            phase_transition(user,data['message'])
            
    elif user.phase == "behavioural":
        if user.question_count < 6:
            ask_behavioural(user, data['message'])
        else:
            phase_transition(user,data['message'])

    elif user.phase == "technical":
        
        if user.question_count < 9:
            ask_technical(user, data['message'])
        else:
            phase_transition(user,data['message'])

@app.route('/extract_features', methods=['POST'])
def extract_features_endpoint():
    """
    Expects JSON: { "participant_id": "P1", "video_filename": "user123_1712345678.webm" }
    """
    data = request.get_json()
    participant_id = data.get("participant_id")
    video_filename = data.get("video_filename")
    if not participant_id or not video_filename:
        return jsonify({"error": "Missing participant_id or video_filename"}), 400

    # Path to the uploaded video file (adjust if needed)
    video_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "Hireverse-Server", "server", "uploads", video_filename)
    )

    if not os.path.exists(video_path):
        return jsonify({"error": "Video file not found"}), 404

    try:
        features = extract_features_from_video(participant_id, video_path)
        return jsonify({"status": "success", "features": str(features)})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == "__main__":
    socketio.run(app)