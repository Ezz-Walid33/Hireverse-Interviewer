from pydoc import text
import string
from flask import Flask, request, jsonify
import pandas as pd
import random
import os
import sys
import base64
sys.path.append(os.path.abspath('./Hireverse-neuroSync/NeuroSync_Player_main'))
from Text_Audio_Face import text_to_wav
wav_dir = os.path.join(os.getcwd(), "Hireverse-neuroSync", "NeuroSync_Player_main", "wav_input")
os.makedirs(wav_dir, exist_ok=True)


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
from utils.generated_runners import run_audio_animation
from livelink.connect.livelink_init import create_socket_connection, initialize_py_face
from livelink.animations.default_animation import default_animation_loop
from inference_from_text import EmotionClassifier


load_dotenv()

app = Flask(__name__)
socketio = SocketIO(app)

# Redirect Flask logs and custom logs to the same file
log = logging.getLogger('werkzeug')
file_handler = logging.FileHandler('flask_logs.log')
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
log.addHandler(file_handler)

skip_blendshapes = True

# Set up a custom logger for candidate and interviewer messages
app_logger = logging.getLogger('app_logger')
app_logger.setLevel(logging.INFO)
app_logger.addHandler(file_handler)

coding_lines = [
    "Have a go at this coding problem.",
    "Let's see how you approach this coding challenge.",
    "Try solving this programming task.",
    "Here's a coding problem for you to tackle.",
    "Give this coding exercise a shot.",
    "Let's work through this coding question together."
]

def get_random_coding_line():
    return random.choice(coding_lines)

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
        DO NOT THANK THE CANDIDATE. Follow the instructions exactly - if told not to ask questions, provide only comments."""),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{input}")
    ])
behavioral_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are conducting a behavioral interview. Ask questions that reveal 
    the candidate's soft skills, problem-solving approach, and cultural fit.
    Ask ONE question at a time and wait for their response. Keep conversations natural 
    by focusing on one topic before moving to the next. When you need follow-up details, 
    ask one specific follow-up question rather than multiple questions at once.
    DONT THANK THE CANDIDATE or use overly positive language. Keep responses professional and focused.
    Follow the instructions exactly - if told not to ask questions, provide only comments."""),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

technical_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are a technical interviewer. Ask questions precisely and 
    evaluate answers professionally. Ask ONE technical question at a time and wait 
    for their response before proceeding to the next question.
    When providing feedback on answers, be brief and constructive without giving away 
    correct answers. Focus on one concept at a time for clear understanding.
    DO NOT THANK THE CANDIDATE. Follow the instructions exactly - if told not to ask questions, provide only comments."""),
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
# conversation_memory = ConversationBufferMemory(return_messages=True)

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
    }
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
    def __init__(self, user_id, name):
        self.user_id = user_id
        self.session_id = generate_id()
        self.name = name
        self.adk_connected = False
        self.phase = PHASES[0]["name"]  # Start with the first phase
        self.question_count = 0
        self.coding_question_asked = False  # Track if coding question has been asked
        # Per-user phase histories
        self.history = {
            "greeting": [],
            "behavioural": [],
            "technical": [],
            "coding": []
        }
        self.eval ={
            "behavioural": {
                "score": None,
                "feedback": None
            },
            "technical": {
                "score": None,
                "feedback": None
            },
            "coding": {
                "score": None,
                "feedback": None
            }
        }
        # Per-user conversation memory
        self.conversation_memory = ConversationBufferMemory(return_messages=True)

    def __repr__(self):
        return f"InterviewUser(user_id={self.user_id})"

def create_app():
    app = Flask(__name__)
    socketio.init_app(app)
    return app
users = []

@socketio.on('connect')
def handle_connect():
    print("Node.js socket connected.")

@socketio.on('disconnect')
def handle_disconnect():
    print('Node.js socket connection lost.')

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
# Initialize these ONCE at startup
py_face = initialize_py_face()
socket_connection = create_socket_connection()
from threading import Thread
default_animation_thread = Thread(target=default_animation_loop, args=(py_face,))
default_animation_thread.start()

def emit_ai_response_with_audio(user, phase, response, is_transition, eval_data=None, text_response=None):
    """
    Generates audio for the AI response, sends it to the neurosync local API,
    emits the ai_response event via socketio (only once, after facial animation starts).
    If eval_data is provided, it will be included in the emit (for end of interview).
    If text_response is provided, it will be used for the text display instead of the audio response.
    """
    try:
        # 1. Generate audio file from text (using response for audio)
        audio_filename = f"{generate_id(12)}.wav"
        audio_path = os.path.join(wav_dir, audio_filename)
        print(f"[INFO] Generated audio file: {audio_path}")
        text_to_wav(response, audio_path)

        # 2. Read audio file as bytes
        with open(audio_path, "rb") as f:
            audio_bytes = f.read()
            audio_b64 = base64.b64encode(audio_bytes).decode('utf-8')

        # 3. Send audio bytes to the neurosync local API
        if(not skip_blendshapes):
            api_url = "http://127.0.0.1:7000/audio_to_blendshapes"
            api_response = requests.post(api_url, data=audio_bytes)
            if api_response.status_code == 200:
                blendshapes = api_response.json().get("blendshapes")
            # Send to Unreal via LiveLink (start animation)
            # Use text_response for display if provided, otherwise use response
                run_audio_animation(audio_bytes, blendshapes, py_face, socket_connection, default_animation_thread)
            # Emit the AI response as before, with optional eval data (only once, after animation starts)
        
        # else:
        #     print(f"[ERROR] Neurosync API error: {api_response.status_code} {api_response.text}")
        #     blendshapes = None
        display_text = text_response if text_response is not None else response
        emit_data = {
            "phase": phase,
            "response": display_text,
            "recipient": user.user_id,
            "audio": audio_b64,
            **({"transition": True} if is_transition else {})
            }
        if eval_data is not None:
            emit_data["eval"] = eval_data
        socketio.emit('ai_response', emit_data)

    except Exception as e:
        print(f"Audio/blendshape generation failed: {e}")
        #blendshapes = None



def phase_transition(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    history = user.conversation_memory.chat_memory.messages
    phase = get_phase(user.phase)
    if not phase:
        print(f"Unknown phase: {user.phase}")
        return

    next_phase_name = get_next_phase_name(user.phase)
    phase_history = user.history.get(user.phase, [])
    eval = evaluation(phase_history, user.phase)
    if user.phase != "greeting":
        user.eval[user.phase] = {
            "score": extract_score_from_evaluation(eval),
            "feedback": eval
        }
        print(colored(f'{user.phase.capitalize()} Phase Evaluation:', 'green'))
        print(colored("Score:", 'light_yellow') + f" {user.eval[user.phase]['score']}")
        print(colored('Feedback:', 'light_cyan') + f" {user.eval[user.phase]['feedback']}")
    if next_phase_name != "end":
        prompt = {
            "input": f'The candidate just said: "{user_input}" in the {user.phase} phase. Comment on it briefly and positively. Do not ask any questions. Just provide a brief acknowledgment and transition comment to wrap up this phase.',
            "history": history
        }
        if user.phase == "coding":
            prompt["assistant"] = []
    else:
        prompt = {
            "input": f'The candidate just said: "{user_input}" in the {user.phase} phase. Comment on it briefly and positively. Do not ask any questions, and segue into the end of the interview.',
            "history": history
        }
        if user.phase == "coding":
            prompt["assistant"] = []

    response = invoke_with_rate_limit(phase["chain"](), prompt, user)
    if response is None:
        return
    print(colored('Interviewer:', 'cyan') + f" {response.content}")
    user.conversation_memory.chat_memory.add_ai_message(response.content)
    if next_phase_name != "end":
        emit_ai_response_with_audio(user, user.phase, response.content, True)
        next_phase = get_next_phase_name(user.phase)
        if next_phase != "end":
            user.phase = next_phase
            user.question_count = 0
            print(colored(f"-----------{next_phase.upper()} PHASE-----------", 'cyan'))
            if next_phase == "coding":
                user.question_count = 1
                ask_coding(user, "Please ask me a coding question")
            else:
                user.question_count = 1
                get_phase(next_phase)["ask_func"](user, f"Let's start the {next_phase} phase")
        else:
            user.phase = "end"
            # End interview emit with audio and eval
            emit_ai_response_with_audio(
                user,
                "end",
                "Thank you for participating in this interview. We will be in touch soon with our decision.",
                False,
                eval_data=user.eval
            )
            # Clear the user's session and remove from users list
            if user in users:
                users.remove(user)
            del user
    else:
        # End interview emit with audio and eval
        emit_ai_response_with_audio(
            user,
            "end",
            "Thank you for participating in this interview. We will be in touch soon with our decision.",
            False,
            eval_data=user.eval
        )
        # Clear the user's session and remove from users list
        if user in users:
            users.remove(user)
        del user

@socketio.on('start_interview')
def handle_start_interview(data):
    print(colored("-----------START INTERVIEW-----------", 'cyan'))
    user = InterviewUser( data['userId'], data['name']) 
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
        socketio.emit('ai_response', {"phase": "error", "response": "A network error occurred. Please try again later.", "recipient": data['userId']})
        return
    response = invoke_with_rate_limit(greeting_chain, {
        "input": "Greet the candidate warmly and ask ONE question to get to know them better.",
        "history": []
    }, user)
    if response is None:
        return
    user.conversation_memory.chat_memory.add_ai_message(str(response.content))
    user.question_count = 1
    user.history["greeting"].append(f"AI: {response.content}")
    print(colored('Interviewer:', 'cyan') + f" {response.content}")
    app_logger.info(f"Interviewer: {response.content}")
    emit_ai_response_with_audio(user, "greeting", response.content, False)



def small_talk(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    user.history["greeting"].append(f"Candidate: {user_input}")

    history = user.conversation_memory.chat_memory.messages

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
    print(colored('Interviewer:', 'cyan') + f" {response.content}")
    user.conversation_memory.chat_memory.add_ai_message(response.content)
    user.history["greeting"].append(f"AI: {response.content}")
    # Never set transition True here
    emit_ai_response_with_audio(user, user.phase, response.content, False)

def ask_behavioural(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    user.history["behavioural"].append(f"Candidate: {user_input}")

    if not hasattr(user, 'behavioral_questions'):
        user.behavioral_questions = random.sample(behavioral_questions, min(3, len(behavioral_questions)))

    history = user.conversation_memory.chat_memory.messages
    question_index = user.question_count - 1

    if question_index < len(user.behavioral_questions):
        question = user.behavioral_questions[question_index]

        if user.question_count == 1:
            prompt_text = f"Ask this behavioral question: {question}"
        else:
            prompt_text = f"Ask ONE follow-up question to get more specific details about their previous answer. If they've provided enough detail, move on to ask: {question}"

        response = invoke_with_rate_limit(behavioral_chain, {
            "input": prompt_text,
            "history": history
        }, user)

        if response is None:
            return

        print(colored('Interviewer:', 'cyan') + f" {response.content}")
        user.conversation_memory.chat_memory.add_ai_message(response.content)
        user.history["behavioural"].append(f"AI: {response.content}")
        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, response.content, False)

def ask_technical(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    user.history["technical"].append(f"Candidate: {user_input}")

    if not hasattr(user, 'technical_questions'):
        user.technical_questions = random.sample(tech_questions, min(3, len(tech_questions)))

    history = user.conversation_memory.chat_memory.messages
    question_index = user.question_count - 1

    if question_index < len(user.technical_questions):
        question_data = user.technical_questions[question_index]
        question = question_data['Question']

        if user.question_count == 1:
            prompt_text = f"Declare that you are now going to ask a few technical questions, then ask this question: {question}"
        else:
            prompt_text = f"Provide brief feedback on their technical answer without giving away correct answers. Then ask: {question}"

        response = invoke_with_rate_limit(technical_chain, {
            "input": prompt_text,
            "history": history
        }, user)

        if response is None:
            return

        print(colored('Interviewer:', 'cyan') + f" {response.content}")
        user.conversation_memory.chat_memory.add_ai_message(response.content)
        user.history["technical"].append(f"AI: {response.content}")
        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, response.content, False)

def ask_coding(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)

    if not user.coding_question_asked:
        adk_response = send_message_to_adk(user, "Please ask the coding question")
        if not adk_response:
            print("No response from ADK")
            return

        user.coding_question_asked = True
        user.conversation_memory.chat_memory.add_ai_message(adk_response)
        user.history["coding"].append(f"AI: {adk_response}")

        print(f"ADK Response: {adk_response}")

        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, adk_response, False)
    else:
        user.history["coding"].append(f"Candidate: {user_input}")

        adk_response = send_message_to_adk(user, user_input)
        if not adk_response:
            print("No response from ADK")
            return

        user.conversation_memory.chat_memory.add_ai_message(adk_response)
        user.history["coding"].append(f"AI: {adk_response}")

        print(f"ADK Evaluation: {adk_response}")

        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, adk_response, False)

def phase_transition(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    history = user.conversation_memory.chat_memory.messages
    phase = get_phase(user.phase)
    if not phase:
        print(f"Unknown phase: {user.phase}")
        return

    next_phase_name = get_next_phase_name(user.phase)
    phase_history = user.history.get(user.phase, [])
    eval = evaluation(phase_history, user.phase)
    if user.phase != "greeting":
        user.eval[user.phase] = {
            "score": extract_score_from_evaluation(eval),
            "feedback": eval
        }
        print(colored(f'{user.phase.capitalize()} Phase Evaluation:', 'green'))
        print(colored("Score:", 'light_yellow') + f" {user.eval[user.phase]['score']}")
        print(colored('Feedback:', 'light_cyan') + f" {user.eval[user.phase]['feedback']}")
    if next_phase_name != "end":
        prompt = {
            "input": f'The candidate just said: "{user_input}" in the {user.phase} phase. Comment on it briefly and positively. Do not ask any questions. Just provide a brief acknowledgment and transition comment to wrap up this phase.',
            "history": history
        }
        if user.phase == "coding":
            prompt["assistant"] = []
    else:
        prompt = {
            "input": f'The candidate just said: "{user_input}" in the {user.phase} phase. Comment on it briefly and positively. Do not ask any questions, and segue into the end of the interview.',
            "history": history
        }
        if user.phase == "coding":
            prompt["assistant"] = []

    response = invoke_with_rate_limit(phase["chain"](), prompt, user)
    if response is None:
        return
    print(colored('Interviewer:', 'cyan') + f" {response.content}")
    user.conversation_memory.chat_memory.add_ai_message(response.content)
    if next_phase_name != "end":
        emit_ai_response_with_audio(user, user.phase, response.content, True)
        next_phase = get_next_phase_name(user.phase)
        if next_phase != "end":
            user.phase = next_phase
            user.question_count = 0
            print(colored(f"-----------{next_phase.upper()} PHASE-----------", 'cyan'))
            if next_phase == "coding":
                user.question_count = 1
                ask_coding(user, "Please ask me a coding question")
            else:
                user.question_count = 1
                get_phase(next_phase)["ask_func"](user, f"Let's start the {next_phase} phase")
        else:
            user.phase = "end"
            # End interview emit with audio and eval
            emit_ai_response_with_audio(
                user,
                "end",
                "Thank you for participating in this interview. We will be in touch soon with our decision.",
                False,
                eval_data=user.eval
            )
            # Clear the user's session and remove from users list
            if user in users:
                users.remove(user)
            del user
    else:
        # End interview emit with audio and eval
        emit_ai_response_with_audio(
            user,
            "end",
            "Thank you for participating in this interview. We will be in touch soon with our decision.",
            False,
            eval_data=user.eval
        )
        # Clear the user's session and remove from users list
        if user in users:
            users.remove(user)
        del user

@socketio.on('start_interview')
def handle_start_interview(data):
    print(colored("-----------START INTERVIEW-----------", 'cyan'))
    user = InterviewUser( data['userId'], data['name']) 
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
        socketio.emit('ai_response', {"phase": "error", "response": "A network error occurred. Please try again later.", "recipient": data['userId']})
        return
    response = invoke_with_rate_limit(greeting_chain, {
        "input": "Greet the candidate warmly and ask ONE question to get to know them better.",
        "history": []
    }, user)
    if response is None:
        return
    user.conversation_memory.chat_memory.add_ai_message(str(response.content))
    user.question_count = 1
    user.history["greeting"].append(f"AI: {response.content}")
    print(colored('Interviewer:', 'cyan') + f" {response.content}")
    app_logger.info(f"Interviewer: {response.content}")
    emit_ai_response_with_audio(user, "greeting", response.content, False)



def small_talk(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    user.history["greeting"].append(f"Candidate: {user_input}")

    history = user.conversation_memory.chat_memory.messages

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
    print(colored('Interviewer:', 'cyan') + f" {response.content}")
    user.conversation_memory.chat_memory.add_ai_message(response.content)
    user.history["greeting"].append(f"AI: {response.content}")
    # Never set transition True here
    emit_ai_response_with_audio(user, user.phase, response.content, False)

def ask_behavioural(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    user.history["behavioural"].append(f"Candidate: {user_input}")

    if not hasattr(user, 'behavioral_questions'):
        user.behavioral_questions = random.sample(behavioral_questions, min(3, len(behavioral_questions)))

    history = user.conversation_memory.chat_memory.messages
    question_index = user.question_count - 1

    if question_index < len(user.behavioral_questions):
        question = user.behavioral_questions[question_index]

        if user.question_count == 1:
            prompt_text = f"Ask this behavioral question: {question}"
        else:
            prompt_text = f"Ask ONE follow-up question to get more specific details about their previous answer. If they've provided enough detail, move on to ask: {question}"

        response = invoke_with_rate_limit(behavioral_chain, {
            "input": prompt_text,
            "history": history
        }, user)

        if response is None:
            return

        print(colored('Interviewer:', 'cyan') + f" {response.content}")
        user.conversation_memory.chat_memory.add_ai_message(response.content)
        user.history["behavioural"].append(f"AI: {response.content}")
        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, response.content, False)

def ask_technical(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    user.history["technical"].append(f"Candidate: {user_input}")

    if not hasattr(user, 'technical_questions'):
        user.technical_questions = random.sample(tech_questions, min(3, len(tech_questions)))

    history = user.conversation_memory.chat_memory.messages
    question_index = user.question_count - 1

    if question_index < len(user.technical_questions):
        question_data = user.technical_questions[question_index]
        question = question_data['Question']

        if user.question_count == 1:
            prompt_text = f"Declare that you are now going to ask a few technical questions, then ask this question: {question}"
        else:
            prompt_text = f"Provide brief feedback on their technical answer without giving away correct answers. Then ask: {question}"

        response = invoke_with_rate_limit(technical_chain, {
            "input": prompt_text,
            "history": history
        }, user)

        if response is None:
            return

        print(colored('Interviewer:', 'cyan') + f" {response.content}")
        user.conversation_memory.chat_memory.add_ai_message(response.content)
        user.history["technical"].append(f"AI: {response.content}")
        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, response.content, False)

def ask_coding(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)

    if not user.coding_question_asked:
        adk_response = send_message_to_adk(user, "Please ask the coding question")
        if not adk_response:
            print("No response from ADK")
            return

        user.coding_question_asked = True
        user.conversation_memory.chat_memory.add_ai_message(adk_response)
        user.history["coding"].append(f"AI: {adk_response}")

        print(f"ADK Response: {adk_response}")

        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, adk_response, False)
    else:
        user.history["coding"].append(f"Candidate: {user_input}")

        adk_response = send_message_to_adk(user, user_input)
        if not adk_response:
            print("No response from ADK")
            return

        user.conversation_memory.chat_memory.add_ai_message(adk_response)
        user.history["coding"].append(f"AI: {adk_response}")

        print(f"ADK Evaluation: {adk_response}")

        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, adk_response, False)

@socketio.on('end_user_session')
def handle_end_user_session(data):
    """Handle session cleanup when Node.js server ends a user session"""
    user_id = data.get('userId')
    if not user_id:
        return
    
    # Find and remove the user from the users list
    global users
    users_to_remove = [user for user in users if user.user_id == user_id]
    for user in users_to_remove:
        users.remove(user)
        print(colored(f"Session cleanup: Removed user {user_id} from Flask server", 'yellow'))
        del user

def phase_transition(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    history = user.conversation_memory.chat_memory.messages
    phase = get_phase(user.phase)
    if not phase:
        print(f"Unknown phase: {user.phase}")
        return

    next_phase_name = get_next_phase_name(user.phase)
    phase_history = user.history.get(user.phase, [])
    eval = evaluation(phase_history, user.phase)
    if user.phase != "greeting":
        user.eval[user.phase] = {
            "score": extract_score_from_evaluation(eval),
            "feedback": eval
        }
        print(colored(f'{user.phase.capitalize()} Phase Evaluation:', 'green'))
        print(colored("Score:", 'light_yellow') + f" {user.eval[user.phase]['score']}")
        print(colored('Feedback:', 'light_cyan') + f" {user.eval[user.phase]['feedback']}")
    if next_phase_name != "end":
        prompt = {
            "input": f'The candidate just said: "{user_input}" in the {user.phase} phase. Comment on it briefly and positively. Do not ask any questions. Just provide a brief acknowledgment and transition comment to wrap up this phase.',
            "history": history
        }
        if user.phase == "coding":
            prompt["assistant"] = []
    else:
        prompt = {
            "input": f'The candidate just said: "{user_input}" in the {user.phase} phase. Comment on it briefly and positively. Do not ask any questions, and segue into the end of the interview.',
            "history": history
        }
        if user.phase == "coding":
            prompt["assistant"] = []

    response = invoke_with_rate_limit(phase["chain"](), prompt, user)
    if response is None:
        return
    print(colored('Interviewer:', 'cyan') + f" {response.content}")
    user.conversation_memory.chat_memory.add_ai_message(response.content)
    if next_phase_name != "end":
        emit_ai_response_with_audio(user, user.phase, response.content, True)
        next_phase = get_next_phase_name(user.phase)
        if next_phase != "end":
            user.phase = next_phase
            user.question_count = 0
            print(colored(f"-----------{next_phase.upper()} PHASE-----------", 'cyan'))
            if next_phase == "coding":
                user.question_count = 1
                ask_coding(user, "Please ask me a coding question")
            else:
                user.question_count = 1
                get_phase(next_phase)["ask_func"](user, f"Let's start the {next_phase} phase")
        else:
            user.phase = "end"
            # End interview emit with audio and eval
            emit_ai_response_with_audio(
                user,
                "end",
                "Thank you for participating in this interview. We will be in touch soon with our decision.",
                False,
                eval_data=user.eval
            )
            # Clear the user's session and remove from users list
            if user in users:
                users.remove(user)
            del user
    else:
        # End interview emit with audio and eval
        emit_ai_response_with_audio(
            user,
            "end",
            "Thank you for participating in this interview. We will be in touch soon with our decision.",
            False,
            eval_data=user.eval
        )
        # Clear the user's session and remove from users list
        if user in users:
            users.remove(user)
        del user

@socketio.on('start_interview')
def handle_start_interview(data):
    print(colored("-----------START INTERVIEW-----------", 'cyan'))
    user = InterviewUser( data['userId'], data['name']) 
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
        socketio.emit('ai_response', {"phase": "error", "response": "A network error occurred. Please try again later.", "recipient": data['userId']})
        return
    response = invoke_with_rate_limit(greeting_chain, {
        "input": "Greet the candidate warmly and ask ONE question to get to know them better.",
        "history": []
    }, user)
    if response is None:
        return
    user.conversation_memory.chat_memory.add_ai_message(str(response.content))
    user.question_count = 1
    user.history["greeting"].append(f"AI: {response.content}")
    print(colored('Interviewer:', 'cyan') + f" {response.content}")
    app_logger.info(f"Interviewer: {response.content}")
    emit_ai_response_with_audio(user, "greeting", response.content, False)



def small_talk(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    user.history["greeting"].append(f"Candidate: {user_input}")

    history = user.conversation_memory.chat_memory.messages

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
    print(colored('Interviewer:', 'cyan') + f" {response.content}")
    user.conversation_memory.chat_memory.add_ai_message(response.content)
    user.history["greeting"].append(f"AI: {response.content}")
    # Never set transition True here
    emit_ai_response_with_audio(user, user.phase, response.content, False)

def ask_behavioural(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    user.history["behavioural"].append(f"Candidate: {user_input}")

    if not hasattr(user, 'behavioral_questions'):
        user.behavioral_questions = random.sample(behavioral_questions, min(3, len(behavioral_questions)))

    history = user.conversation_memory.chat_memory.messages
    question_index = user.question_count - 1

    if question_index < len(user.behavioral_questions):
        question = user.behavioral_questions[question_index]

        if user.question_count == 1:
            prompt_text = f"Ask this behavioral question: {question}"
        else:
            prompt_text = f"Ask ONE follow-up question to get more specific details about their previous answer. If they've provided enough detail, move on to ask: {question}"

        response = invoke_with_rate_limit(behavioral_chain, {
            "input": prompt_text,
            "history": history
        }, user)

        if response is None:
            return

        print(colored('Interviewer:', 'cyan') + f" {response.content}")
        user.conversation_memory.chat_memory.add_ai_message(response.content)
        user.history["behavioural"].append(f"AI: {response.content}")
        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, response.content, False)

def ask_technical(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    user.history["technical"].append(f"Candidate: {user_input}")

    if not hasattr(user, 'technical_questions'):
        user.technical_questions = random.sample(tech_questions, min(3, len(tech_questions)))

    history = user.conversation_memory.chat_memory.messages
    question_index = user.question_count - 1

    if question_index < len(user.technical_questions):
        question_data = user.technical_questions[question_index]
        question = question_data['Question']

        if user.question_count == 1:
            prompt_text = f"Declare that you are now going to ask a few technical questions, then ask this question: {question}"
        else:
            prompt_text = f"Provide brief feedback on their technical answer without giving away correct answers. Then ask: {question}"

        response = invoke_with_rate_limit(technical_chain, {
            "input": prompt_text,
            "history": history
        }, user)

        if response is None:
            return

        print(colored('Interviewer:', 'cyan') + f" {response.content}")
        user.conversation_memory.chat_memory.add_ai_message(response.content)
        user.history["technical"].append(f"AI: {response.content}")
        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, response.content, False)

def ask_coding(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)

    if not user.coding_question_asked:
        adk_response = send_message_to_adk(user, "Please ask the coding question")
        if not adk_response:
            print("No response from ADK")
            return

        user.coding_question_asked = True
        user.conversation_memory.chat_memory.add_ai_message(adk_response)
        user.history["coding"].append(f"AI: {adk_response}")

        print(f"ADK Response: {adk_response}")

        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, adk_response, False)
    else:
        user.history["coding"].append(f"Candidate: {user_input}")

        adk_response = send_message_to_adk(user, user_input)
        if not adk_response:
            print("No response from ADK")
            return

        user.conversation_memory.chat_memory.add_ai_message(adk_response)
        user.history["coding"].append(f"AI: {adk_response}")

        print(f"ADK Evaluation: {adk_response}")

        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, adk_response, False)

@socketio.on('end_user_session')
def handle_end_user_session(data):
    """Handle session cleanup when Node.js server ends a user session"""
    user_id = data.get('userId')
    if not user_id:
        return
    
    # Find and remove the user from the users list
    global users
    users_to_remove = [user for user in users if user.user_id == user_id]
    for user in users_to_remove:
        users.remove(user)
        print(colored(f"Session cleanup: Removed user {user_id} from Flask server", 'yellow'))
        del user

def phase_transition(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    history = user.conversation_memory.chat_memory.messages
    phase = get_phase(user.phase)
    if not phase:
        print(f"Unknown phase: {user.phase}")
        return

    next_phase_name = get_next_phase_name(user.phase)
    phase_history = user.history.get(user.phase, [])
    eval = evaluation(phase_history, user.phase)
    if user.phase != "greeting":
        user.eval[user.phase] = {
            "score": extract_score_from_evaluation(eval),
            "feedback": eval
        }
        print(colored(f'{user.phase.capitalize()} Phase Evaluation:', 'green'))
        print(colored("Score:", 'light_yellow') + f" {user.eval[user.phase]['score']}")
        print(colored('Feedback:', 'light_cyan') + f" {user.eval[user.phase]['feedback']}")
    if next_phase_name != "end":
        prompt = {
            "input": f'The candidate just said: "{user_input}" in the {user.phase} phase. Comment on it briefly and positively. Do not ask any questions. Just provide a brief acknowledgment and transition comment to wrap up this phase.',
            "history": history
        }
        if user.phase == "coding":
            prompt["assistant"] = []
    else:
        prompt = {
            "input": f'The candidate just said: "{user_input}" in the {user.phase} phase. Comment on it briefly and positively. Do not ask any questions, and segue into the end of the interview.',
            "history": history
        }
        if user.phase == "coding":
            prompt["assistant"] = []

    response = invoke_with_rate_limit(phase["chain"](), prompt, user)
    if response is None:
        return
    print(colored('Interviewer:', 'cyan') + f" {response.content}")
    user.conversation_memory.chat_memory.add_ai_message(response.content)
    if next_phase_name != "end":
        emit_ai_response_with_audio(user, user.phase, response.content, True)
        next_phase = get_next_phase_name(user.phase)
        if next_phase != "end":
            user.phase = next_phase
            user.question_count = 0
            print(colored(f"-----------{next_phase.upper()} PHASE-----------", 'cyan'))
            if next_phase == "coding":
                user.question_count = 1
                ask_coding(user, "Please ask me a coding question")
            else:
                user.question_count = 1
                get_phase(next_phase)["ask_func"](user, f"Let's start the {next_phase} phase")
        else:
            user.phase = "end"
            # End interview emit with audio and eval
            emit_ai_response_with_audio(
                user,
                "end",
                "Thank you for participating in this interview. We will be in touch soon with our decision.",
                False,
                eval_data=user.eval
            )
            # Clear the user's session and remove from users list
            if user in users:
                users.remove(user)
            del user
    else:
        # End interview emit with audio and eval
        emit_ai_response_with_audio(
            user,
            "end",
            "Thank you for participating in this interview. We will be in touch soon with our decision.",
            False,
            eval_data=user.eval
        )
        # Clear the user's session and remove from users list
        if user in users:
            users.remove(user)
        del user

@socketio.on('start_interview')
def handle_start_interview(data):
    print(colored("-----------START INTERVIEW-----------", 'cyan'))
    user = InterviewUser( data['userId'], data['name']) 
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
        socketio.emit('ai_response', {"phase": "error", "response": "A network error occurred. Please try again later.", "recipient": data['userId']})
        return
    response = invoke_with_rate_limit(greeting_chain, {
        "input": "Greet the candidate warmly and ask ONE question to get to know them better.",
        "history": []
    }, user)
    if response is None:
        return
    user.conversation_memory.chat_memory.add_ai_message(str(response.content))
    user.question_count = 1
    user.history["greeting"].append(f"AI: {response.content}")
    print(colored('Interviewer:', 'cyan') + f" {response.content}")
    app_logger.info(f"Interviewer: {response.content}")
    emit_ai_response_with_audio(user, "greeting", response.content, False)



def small_talk(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    user.history["greeting"].append(f"Candidate: {user_input}")

    history = user.conversation_memory.chat_memory.messages

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
    print(colored('Interviewer:', 'cyan') + f" {response.content}")
    user.conversation_memory.chat_memory.add_ai_message(response.content)
    user.history["greeting"].append(f"AI: {response.content}")
    # Never set transition True here
    emit_ai_response_with_audio(user, user.phase, response.content, False)

def ask_behavioural(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    user.history["behavioural"].append(f"Candidate: {user_input}")

    if not hasattr(user, 'behavioral_questions'):
        user.behavioral_questions = random.sample(behavioral_questions, min(3, len(behavioral_questions)))

    history = user.conversation_memory.chat_memory.messages
    question_index = user.question_count - 1

    if question_index < len(user.behavioral_questions):
        question = user.behavioral_questions[question_index]

        if user.question_count == 1:
            prompt_text = f"Ask this behavioral question: {question}"
        else:
            prompt_text = f"Ask ONE follow-up question to get more specific details about their previous answer. If they've provided enough detail, move on to ask: {question}"

        response = invoke_with_rate_limit(behavioral_chain, {
            "input": prompt_text,
            "history": history
        }, user)

        if response is None:
            return

        print(colored('Interviewer:', 'cyan') + f" {response.content}")
        user.conversation_memory.chat_memory.add_ai_message(response.content)
        user.history["behavioural"].append(f"AI: {response.content}")
        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, response.content, False)

def ask_technical(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)
    user.history["technical"].append(f"Candidate: {user_input}")

    if not hasattr(user, 'technical_questions'):
        user.technical_questions = random.sample(tech_questions, min(3, len(tech_questions)))

    history = user.conversation_memory.chat_memory.messages
    question_index = user.question_count - 1

    if question_index < len(user.technical_questions):
        question_data = user.technical_questions[question_index]
        question = question_data['Question']

        if user.question_count == 1:
            prompt_text = f"Declare that you are now going to ask a few technical questions, then ask this question: {question}"
        else:
            prompt_text = f"Provide brief feedback on their technical answer without giving away correct answers. Then ask: {question}"

        response = invoke_with_rate_limit(technical_chain, {
            "input": prompt_text,
            "history": history
        }, user)

        if response is None:
            return

        print(colored('Interviewer:', 'cyan') + f" {response.content}")
        user.conversation_memory.chat_memory.add_ai_message(response.content)
        user.history["technical"].append(f"AI: {response.content}")
        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, response.content, False)

def ask_coding(user, user_input):
    user.conversation_memory.chat_memory.add_user_message(user_input)

    if not user.coding_question_asked:
        adk_response = send_message_to_adk(user, "Please ask the coding question")
        if not adk_response:
            print("No response from ADK")
            return

        user.coding_question_asked = True
        user.conversation_memory.chat_memory.add_ai_message(adk_response)
        user.history["coding"].append(f"AI: {adk_response}")

        print(f"ADK Response: {adk_response}")

        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, adk_response, False)
    else:
        user.history["coding"].append(f"Candidate: {user_input}")

        adk_response = send_message_to_adk(user, user_input)
        if not adk_response:
            print("No response from ADK")
            return

        user.conversation_memory.chat_memory.add_ai_message(adk_response)
        user.history["coding"].append(f"AI: {adk_response}")

        print(f"ADK Evaluation: {adk_response}")

        # Never set transition True here
        emit_ai_response_with_audio(user, user.phase, adk_response, False)

@socketio.on('end_user_session')
def handle_end_user_session(data):
    """Handle session cleanup when Node.js server ends a user session"""
    user_id = data.get('userId')
    if not user_id:
        return
    
    # Find and remove the user from the users list
    global users
    users_to_remove = [user for user in users if user.user_id == user_id]
    for user in users_to_remove:
        users.remove(user)
        print(colored(f"Session cleanup: Removed user {user_id} from Flask server", 'yellow'))
        del user

@socketio.on('message')
def handle_message(data):
    user_id = data.get('userId')
    user = None
    for u in users:
        if u.user_id == user_id:
            user = u
            break
    if user is None:
        print(f"[ERROR] No user found for userId: {user_id}")
        app_logger.error(f"No user found for userId: {user_id}")
        return

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


    if user.question_count <= phase["question_limit"]:
        phase["ask_func"](user, data['message'])
    else:
        # Transition to next phase
        phase_transition(user, data['message'])

@socketio.on('end_interview')
def handle_end_interview(data):
    user_id = data.get('userId')
    user = None
    for u in users:
        if u.user_id == user_id:
            user = u
            break
    if user is None:
        print(f"[ERROR] No user found for userId: {user_id}")
        app_logger.error(f"No user found for userId: {user_id}")
        return

    print(colored(f"Ending interview for {user.name}", 'cyan'))
    app_logger.info(f"Ending interview for {user.name}")

    # Clear the user's session and remove from users list
    if user in users:
        users.remove(user)
    del user

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
        return jsonify({"status": "success", "instances": features})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route('/extract_emotion', methods=['POST'])
def extract_emotion_endpoint():
    """
    Emotion extraction endpoint for interview analysis
    Expects JSON: { "messages": ["message1", "message2", ...] }
    Returns emotion analysis for the provided messages
    """
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
        
        messages = data.get("messages", [])
        
        if not messages:
            return jsonify({"error": "No messages provided"}), 400
        
        if not isinstance(messages, list):
            return jsonify({"error": "Messages must be an array"}), 400
        
        # Filter out empty messages
        valid_messages = [msg for msg in messages if msg and isinstance(msg, str) and msg.strip()]
        
        if not valid_messages:
            return jsonify({"error": "No valid messages found"}), 400
        
        # Initialize emotion classifier
        classifier = EmotionClassifier()
        
        # Get emotion breakdown for database storage
        emotion_breakdown = classifier.get_emotion_breakdown_for_database(valid_messages)
        
        if "error" in emotion_breakdown:
            return jsonify({"error": emotion_breakdown["error"]}), 500
        
        return jsonify({
            "status": "success",
            "emotion_analysis": emotion_breakdown
        })
        
    except Exception as e:
        app_logger.error(f"Error in emotion extraction: {str(e)}")
        return jsonify({"error": f"Emotion extraction failed: {str(e)}"}), 500


if __name__ == "__main__":
    socketio.run(app, allow_unsafe_werkzeug=True)