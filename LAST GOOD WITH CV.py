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
import logging

load_dotenv()

app = Flask(__name__)

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

# Flask routes
@app.route('/start_interview', methods=['GET'])
def start_interview():
    
    response = greeting_chain.invoke({
        "input": "Greet the candidate warmly.",
        "history": []
    })
    conversation_memory.chat_memory.add_user_message("Greet the candidate warmly")
    conversation_memory.chat_memory.add_ai_message(str(response.content))
    
    # Log the AI's response to the console and to the log file
    print(f"Interviewer: {response.content}")
    app_logger.info(f"Interviewer: {response.content}")
    
    return jsonify({
        "phase": "greeting",
        "response": response.content
    })

@app.route('/ask_question', methods=['POST'])
def small_talk():
   user_input = input("Candidate: ")
   conversation_memory.chat_memory.add_user_message(user_input)
        
   history = conversation_memory.chat_memory.messages
   response = greeting_chain.invoke({
    "input": user_input,
    "history": history
  })
   print(f"\nInterviewer: {response.content}")
   conversation_memory.chat_memory.add_ai_message(response.content) 
   return jsonify({
        "response": response.content
    })


@app.route('/ask_question', methods=['POST'])
def ask_behavioural():
    selected_behavioral = random.sample(behavioral_questions, min(3, len(behavioral_questions)))
    
    for question in selected_behavioral:
        history = conversation_memory.chat_memory.messages
        response = behavioral_chain.invoke({
            "input": f"""Comment briefly on the last thing they said and then ask: {question}
            Don't mention the interview process or thank them.""",
            "history": history
        })
        print(f"\nInterviewer: {response.content}")
        conversation_memory.chat_memory.add_ai_message(response.content)
        
        candidate_answer = input("Candidate: ")
        conversation_memory.chat_memory.add_user_message(candidate_answer)
        
        # Follow-up
        history = conversation_memory.chat_memory.messages
        follow_up = behavioral_chain.invoke({
            "input": f"Generate one relevant follow-up based on: {candidate_answer}",
            "history": history
        })
        print(f"\nInterviewer: {follow_up.content}")
        conversation_memory.chat_memory.add_ai_message(follow_up.content)
        
        candidate_followup = input("Candidate: ")
        conversation_memory.chat_memory.add_user_message(candidate_followup)


@app.route('/ask_question', methods=['POST'])
def ask_technical():
    selected_technical = random.sample(tech_questions, 3)
    
    for question_data in selected_technical:
        question = question_data['Question']
        model_answer = question_data['Answer']
        
        history = conversation_memory.chat_memory.messages
        response = technical_chain.invoke({
            "input": f"Ask this technical question: {question}",
            "history": history
        })
        print(f"\nInterviewer: {response.content}")
        conversation_memory.chat_memory.add_ai_message(response.content)
        
        candidate_answer = input("Candidate: ")
        conversation_memory.chat_memory.add_user_message(candidate_answer)
        
        evaluation = technical_chain.invoke({
            "input": f"""Evaluate this answer for '{question}':
            Expected: {model_answer}
            Candidate: {candidate_answer}
            Provide specific feedback and score from 1-10""",
            "history": history
        })
        print(f"\nEvaluation: {evaluation.content}")
    
        return jsonify({
            "response": response.content,
            "evaluation": evaluation.content
        })

if __name__ == "__main__":
    app.run(debug=True)