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
from termcolor import colored
import time
from feature_extractor import extract_features_from_video
from send_message_to_ADK import send_message_to_adk #New function for sending messages to ADK
import traceback
import json
import re


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
from langchain_groq import ChatGroq
greeting_llm = ChatGroq(model_name="mistral-saba-24b", temperature=0.7)
behavioral_llm = ChatGroq(model_name="mistral-saba-24b", temperature=0.7)
technical_llm = ChatGroq(model_name="mistral-saba-24b", temperature=0.7)
evaluation_llm = ChatGroq(model_name="mistral-saba-24b", temperature=0.3)

# Create prompt templates
def get_greeting_prompt(cv_text):
    return ChatPromptTemplate.from_messages([
        ("system", f"""You are a professional interviewer conducting a job interview. 
        The candidate's CV contains this information:
        {cv_text}
        Maintain a friendly but professional tone. Ask ONE question at a time and wait 
        for their response before proceeding. Keep responses concise and conversational.
        DO NOT THANK THE CANDIDATE"""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
behavioral_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are conducting a behavioral interview. Ask questions that reveal 
    the candidate's soft skills, problem-solving approach, and cultural fit.
    Ask ONE question at a time and wait for their response. Keep conversations natural 
    by focusing on one topic before moving to the next. When you need follow-up details, 
    ask one specific follow-up question rather than multiple questions at once.
    DONT THANK THE CANDIDATE or use overly positive language. Keep responses professional and focused."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

technical_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a technical interviewer. Ask questions precisely and 
    evaluate answers professionally. Ask ONE technical question at a time and wait 
    for their response before proceeding to the next question.
    When providing feedback on answers, be brief and constructive without giving away 
    correct answers. Focus on one concept at a time for clear understanding.
    DO NOT THANK THE CANDIDATE."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

coding_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a coding interviewer conducting a technical interview. You have an assistant that provides coding problems and evaluates candidate solutions.

IMPORTANT: 
- Always use EXACTLY what the assistant provides. Do not create your own questions or modify the assistant's content.
- DO NOT mention the assistant or reference "[ADK Assistant]" in your responses.
- Present the content as if it's coming directly from you as the interviewer.

When presenting a question: Present the assistant's exact coding question without modification.
When evaluating answers: Use the assistant's exact feedback without modification.

Assistant's input:"""),
    
    MessagesPlaceholder(variable_name="assistant"),

    ("system", "Previous conversation:"), 

    MessagesPlaceholder(variable_name="history"),

    ("human", "{input}")
])

evaluation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an expert interview evaluator. Analyze the conversation from a specific interview phase and provide a comprehensive evaluation.

Your evaluation should include:
1. Overall performance assessment (Excellent/Good/Average/Poor)
2. Key strengths demonstrated by the candidate
3. Areas for improvement
4. Specific examples from the conversation to support your assessment
5. Recommendations for the candidate's development

Be objective, constructive, and specific in your feedback. Focus on communication skills, technical knowledge (if applicable), problem-solving approach, and overall interview performance in this phase.

IMPORTANT: You MUST end your response with EXACTLY this format on the last line:
SCORE: [number from 1-100]

Example:
SCORE: 75"""),
    ("human", """Please evaluate the following interview conversation from the {phase_name} phase:

{conversation_history}

Remember to end with the exact format: SCORE: [number from 1-100]""")
])

# Extract CV text and load questions    
cv_path = os.path.join("Interview Files", "candidate_cv.pdf")  # Adjusted to the Interview Files directory
csv_path = os.path.join("Interview Files", "Software Questions.csv")  # Adjusted to the Interview Files directory
cv_text = extract_cv_text(cv_path)

greeting_chain = get_greeting_prompt(cv_text) | greeting_llm
behavioral_chain = behavioral_prompt | behavioral_llm
technical_chain = technical_prompt | technical_llm
coding_chain = coding_prompt | technical_llm
evaluation_chain = evaluation_prompt | evaluation_llm

# Phase-specific conversation history storage
greeting_phase_history = []
behavioral_phase_history = []
technical_phase_history = []
coding_phase_history = []








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

# Define the phases as an array for easy reordering
PHASES = [
    {
        "name": "greeting",
        "chain": lambda: greeting_chain,
        "question_limit": 3,
        "ask_func": lambda user, msg: small_talk(user, msg),
    },
    {
        "name": "behavioural",
        "chain": lambda: behavioral_chain,
        "question_limit": 3,
        "ask_func": lambda user, msg: ask_behavioural(user, msg),
    },
    {
        "name": "technical",
        "chain": lambda: technical_chain,
        "question_limit": 3,
        "ask_func": lambda user, msg: ask_technical(user, msg),
    },
    {
        "name": "coding",
        "chain": lambda: coding_chain,
        "question_limit": 3,
        "ask_func": lambda user, msg: ask_coding(user, msg),
    },
]

# Helper to get phase index and phase object by name
def get_phase_index(phase_name):
    for idx, phase in enumerate(PHASES):
        if phase["name"] == phase_name:
            return idx
    return None

def get_phase(phase_name):
    idx = get_phase_index(phase_name)
    return PHASES[idx] if idx is not None else None

def get_next_phase_name(current_phase_name):
    idx = get_phase_index(current_phase_name)
    if idx is not None and idx + 1 < len(PHASES):
        return PHASES[idx + 1]["name"]
    return "end"

# Update InterviewUser to track phase by name
class InterviewUser:
    def __init__(self, socket_id, user_id, name):
        self.socket_id = socket_id
        self.user_id = user_id
        self.session_id = generate_id()
        self.name = name
        self.adk_connected = False
        self.phase = PHASES[0]["name"]  # Start with the first phase
        self.question_count = 0
        self.coding_question_asked = False  # Track if coding question has been asked

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

def invoke_with_rate_limit(chain, input_data, user=None):
    try:
        time.sleep(1)  # Add a 1-second delay between requests
        return chain.invoke(input_data)
    except Exception as e:
        error_msg = f"Chain invocation error: {e}\n{traceback.format_exc()}"
        print(error_msg)
        app_logger.error(error_msg)
        # Try to extract JSON error object from the exception string
        error_obj = None
        match = re.search(r'(\{.*\})', str(e))
        if match:
            try:
                error_obj = json.loads(match.group(1))
            except Exception:
                error_obj = match.group(1)  # fallback: send the JSON string
        else:
            error_obj = str(e)
        # Emit only the error object
        if user is not None and hasattr(user, "user_id"):
            socketio.emit(
                "flask_server_error",
                {
                    "response": error_obj,
                    "recipient": user.user_id,
                },
            )
            print(f"Error emitted to user_id {user.user_id}")
        return None

def evaluation(phase_history, phase_name):
    if not phase_history:
        return f"No conversation data available for {phase_name} phase evaluation."
    
    conversation_text = "\n".join(phase_history)
    
    try:
        response = evaluation_chain.invoke({
            "phase_name": phase_name,
            "conversation_history": conversation_text
        })
        return response.content
    except Exception as e:
        error_msg = f"Error evaluating {phase_name} phase: {e}"
        print(error_msg)
        return error_msg

def extract_score_from_evaluation(evaluation_text):
    try:
        import re
        pattern = r'SCORE:\s*(\d+)'
        match = re.search(pattern, evaluation_text)
        if match:
            score = int(match.group(1))

            if 1 <= score <= 100:
                return score
        return None
    except Exception as e:
        print(f"Error extracting score: {e}")
        return None


@socketio.on('start_interview')
def handle_start_interview(data):
    print(colored("-----------START INTERVIEW-----------", 'cyan'))
    print(data)
    user = InterviewUser(data['socketId'], data['userId'], data['name'])
    users.append(user)
    try:
        response = requests.post(url=f"http://localhost:8000/apps/CODEEVAL/users/{data['userId']}/sessions/{user.session_id}", json={})
        if response.status_code == 200:
            user.adk_connected = True
            print(f"ADK connected for user {data['userId']} with session ID {user.session_id}")
        else:
            print(f"User {data['userId']} failed to connect to ADK. Status code {response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"Request exception: {e}")
        app_logger.error(f"Request exception: {e}")
        socketio.emit('ai_response', {"phase": "error", "response": "A network error occurred. Please try again later.", "recipient": data['socketId']})
        return
    response = invoke_with_rate_limit(greeting_chain, {
        "input": "Greet the candidate warmly and ask ONE question to get to know them better.",
        "history": []
    }, user)
    if response is None:
        return
    conversation_memory.chat_memory.add_ai_message(str(response.content))
    # Increment question count since we asked the first question
    user.question_count = 1
    # Store initial AI greeting in greeting phase history
    greeting_phase_history.append(f"AI: {response.content}")
    print(f"{colored('Interviewer:', 'cyan')} {response.content}")
    app_logger.info(f"Interviewer: {response.content}")
    socketio.emit('ai_response', {"phase": "greeting", "response": response.content, "recipient": data['socketId']})


def small_talk(user, user_input):
    conversation_memory.chat_memory.add_user_message(user_input)
    # Store user message in greeting phase history
    greeting_phase_history.append(f"Candidate: {user_input}")
    
    history = conversation_memory.chat_memory.messages
    
    # Tailor response based on question count in greeting phase
    if user.question_count == 1:
        prompt_text = "Respond warmly to their answer and ask ONE follow-up question to learn more about their background or interests."
    elif user.question_count == 2:
        prompt_text = "Comment positively on what they shared and ask ONE more question about their experiences or goals."
    else:
        prompt_text = "Acknowledge what they said and make a brief transition comment about moving to the next part of the interview."
    
    response = invoke_with_rate_limit(greeting_chain, {
        "input": prompt_text,
        "history": history
    }, user)
    
    if response is None:
        return
        
    print(f"{response.content}")
    conversation_memory.chat_memory.add_ai_message(response.content)
    # Store AI response in greeting phase history
    greeting_phase_history.append(f"AI: {response.content}")
    socketio.emit('ai_response', { "phase": user.phase, "response": response.content, "recipient": user.socket_id})

def ask_behavioural(user, user_input):
    # Add user input to conversation history
    conversation_memory.chat_memory.add_user_message(user_input)
    # Store user message in behavioral phase history
    behavioral_phase_history.append(f"Candidate: {user_input}")
    
    # Initialize behavioral questions if not already done
    if not hasattr(user, 'behavioral_questions'):
        user.behavioral_questions = random.sample(behavioral_questions, min(3, len(behavioral_questions)))
    
    history = conversation_memory.chat_memory.messages
    
    # Determine which question to ask based on question count
    question_index = user.question_count - 1
    
    if question_index < len(user.behavioral_questions):
        question = user.behavioral_questions[question_index]
        
        if user.question_count == 1:
            # First behavioral question - just ask the main question
            prompt_text = f"Comment briefly on the last thing the candidate said and then ask this behavioral question: {question}"
        else:
            # Subsequent questions - ask ONE follow-up to get more details, then move to next question
            prompt_text = f"Ask ONE follow-up question to get more specific details about their previous answer. If they've provided enough detail, move on to ask: {question}"
            
        response = invoke_with_rate_limit(behavioral_chain, {
            "input": prompt_text,
            "history": history
        }, user)
        
        if response is None:
            return
            
        print(f"{colored('Interviewer:', 'cyan')} {response.content}")
        conversation_memory.chat_memory.add_ai_message(response.content)
        # Store AI response in behavioral phase history
        behavioral_phase_history.append(f"AI: {response.content}")
        socketio.emit('ai_response', {
            "phase": user.phase,
            "response": response.content,
            "recipient": user.socket_id
        })

def ask_technical(user, user_input):
    print(evaluation(behavioral_phase_history, "technical"))
    print(f"'AAAAAAAAAAAAAAAA======',{greeting_phase_history},'========--=-=-=',{behavioral_phase_history},'========--=-=-=',{technical_phase_history})")
    # Add user input to conversation history
    conversation_memory.chat_memory.add_user_message(user_input)
    # Store user message in technical phase history
    technical_phase_history.append(f"Candidate: {user_input}")
    
    # Initialize technical questions if not already done
    if not hasattr(user, 'technical_questions'):
        user.technical_questions = random.sample(tech_questions, min(3, len(tech_questions)))
    
    history = conversation_memory.chat_memory.messages
    
    # Determine which question to ask based on question count
    question_index = user.question_count - 1
    
    if question_index < len(user.technical_questions):
        question_data = user.technical_questions[question_index]
        question = question_data['Question']
        
        if user.question_count == 1:
            # First technical question - just ask the main question
            prompt_text = f"Declare that you are now going to ask a few technical questions, then ask this question: {question}"
        else:
            # Subsequent questions - provide brief feedback and ask next question
            prompt_text = f"Provide brief feedback on their technical answer without giving away correct answers. Then ask: {question}"
            
        response = invoke_with_rate_limit(technical_chain, {
            "input": prompt_text,
            "history": history
        }, user)
        
        if response is None:
            return
            
        print(f"{colored('Interviewer:', 'cyan')} {response.content}")
        conversation_memory.chat_memory.add_ai_message(response.content)
        # Store AI response in technical phase history
        technical_phase_history.append(f"AI: {response.content}")
        socketio.emit('ai_response', {
            "phase": user.phase,
            "response": response.content,
            "recipient": user.socket_id
        })

def ask_coding(user, user_input):
    # Add the user's input to conversation history
    conversation_memory.chat_memory.add_user_message(user_input)
    
    # If this is the first time in coding phase, ask for a coding question
    if not user.coding_question_asked:
        
        adk_response = send_message_to_adk(user, "Please ask the coding question")
        if not adk_response:
            print("No response from ADK")
            return
        
        user.coding_question_asked = True
        
        # Send ADK response directly to the user
        conversation_memory.chat_memory.add_ai_message(adk_response)
        # Store AI response in coding phase history
        coding_phase_history.append(f"AI: {adk_response}")
        
        print(f"ADK Response: {adk_response}")
        
        socketio.emit('ai_response', { "phase": user.phase, "response": adk_response, "recipient": user.socket_id})
    
    else:
        # Store user message in coding phase history
        coding_phase_history.append(f"Candidate: {user_input}")
        
        # Send the user's answer to ADK for evaluation
        adk_response = send_message_to_adk(user, user_input)
        if not adk_response:
            print("No response from ADK")
            return
        
        conversation_memory.chat_memory.add_ai_message(adk_response)
        coding_phase_history.append(f"AI: {adk_response}")
        
        print(f"ADK Evaluation: {adk_response}")
        
        # Send the evaluation response to the user
        socketio.emit('ai_response', { "phase": user.phase, "response": adk_response, "recipient": user.socket_id})



def phase_transition(user, user_input):
    # Add user input to conversation history
    conversation_memory.chat_memory.add_user_message(user_input)
    
    history = conversation_memory.chat_memory.messages
    phase = get_phase(user.phase)
    if not phase:
        print(f"Unknown phase: {user.phase}")
        return

    # Generate a comment on the last response in this phase
    prompt = {
        "input": f"The candidate just said: \"{user_input}\" in the {user.phase} phase. Comment on it briefly and positively. Do not ask any questions.",
        "history": history
    }
    response = invoke_with_rate_limit(phase["chain"](), prompt, user)
    if response is None:
        return
    print(f"{colored('Interviewer:', 'cyan')} {response.content}")
    conversation_memory.chat_memory.add_ai_message(response.content)
    socketio.emit('ai_response', {"phase": user.phase, "response": response.content, "recipient": user.socket_id})

    # Move to next phase if available
    next_phase = get_next_phase_name(user.phase)
    if next_phase != "end":
        user.phase = next_phase
        user.question_count = 0  # Reset question count for new phase
        print(colored(f"-----------{next_phase.upper()} PHASE-----------", 'cyan'))
        
        # Special handling for coding phase transition
        if next_phase == "coding":
            # For coding phase, we need to ask for the first coding question
            user.question_count = 1  # Set to 1 since we're starting the phase
            ask_coding(user, "Please ask me a coding question")
        else:
            # For other phases, use the normal ask function with a greeting message
            user.question_count = 1  # Set to 1 since we're starting the phase  
            get_phase(next_phase)["ask_func"](user, f"Let's start the {next_phase} phase")
    else:
        user.phase = "end"
        # Interview completed
        socketio.emit('ai_response', {
            "phase": "end", 
            "response": "Thank you for participating in this interview. We will be in touch soon with our decision.",
            "recipient": user.socket_id
        })

@socketio.on('message')
def handle_message(data):
    user = get_user_by_socket_id(data['socketId'])
    print(f"{data} ::: {user}")
    print(colored(f"{user.name}: ", "yellow") + data['message'])
    app_logger.info(f"{user.name}: {data['message']}")
    
    # Skip processing if interview is ended
    if user.phase == "end":
        return
    
    user.question_count += 1

    phase = get_phase(user.phase)
    if not phase:
        print(f"Unknown phase: {user.phase}")
        return

    print(f"Phase: {user.phase}, Question count: {user.question_count}, Limit: {phase['question_limit']}")

    if user.question_count <= phase["question_limit"]:
        phase["ask_func"](user, data['message'])
    else:
        # Transition to next phase
        phase_transition(user, data['message'])

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