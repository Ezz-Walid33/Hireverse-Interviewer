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

load_dotenv()

app = Flask(__name__)

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
greeting_llm = ChatMistralAI(model="mistral-large-latest", temperature=0.7)
behavioral_llm = ChatMistralAI(model="mistral-large-latest", temperature=0.6)
technical_llm = ChatMistralAI(model="mistral-large-latest", temperature=0.5)

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
    cv_path = os.path.join("Interview Files", "candidate_cv.pdf")  # Adjusted to the Interview Files directory
    csv_path = os.path.join("Interview Files", "Software Questions.csv")  # Adjusted to the Interview Files directory
    
    cv_text = extract_cv_text(cv_path)
    tech_questions = load_technical_questions(csv_path)
    behavioral_questions = [
        "Tell me about a time you faced a difficult challenge at work and how you handled it",
        "Describe a situation where you had to work with a difficult team member",
        "Give an example of when you took initiative to improve a process"
    ]
    
    memory = ConversationBufferMemory(return_messages=True)
    
    # Greeting Phase
    greeting_chain = get_greeting_prompt(cv_text) | greeting_llm
    response = greeting_chain.invoke({
        "input": "Greet the candidate warmly.",
        "history": []
    })
    memory.chat_memory.add_user_message("Greet the candidate warmly")
    memory.chat_memory.add_ai_message(response.content)
    
    # Log the AI's response to the console
    print(f"Interviewer: {response.content}")
    
    return jsonify({
        "phase": "greeting",
        "response": response.content,
        "history": serialize_messages(memory.chat_memory.messages)
    })

@app.route('/ask_question', methods=['POST'])
def ask_question():
    data = request.json
    user_input = data.get('user_input', "")  # Only the user's response is required
    
    # Log the user's input to the console
    print(f"User: {user_input}")
    
    # Default to behavioral phase
    phase = "behavioral"
    
    if phase == "behavioral":
        # Behavioral logic
        behavioral_chain = behavioral_prompt | behavioral_llm
        response = behavioral_chain.invoke({
            "input": user_input,  # Pass only the latest user input
            "history": conversation_memory.chat_memory.messages  # Use history as context
        })
    elif phase == "technical":
        # Technical logic (if needed in the future)
        technical_chain = technical_prompt | technical_llm
        response = technical_chain.invoke({
            "input": user_input,  # Pass only the latest user input
            "history": conversation_memory.chat_memory.messages  # Use history as context
        })
    else:
        return jsonify({"error": "Invalid phase"}), 400
    
    # Update the conversation memory
    conversation_memory.chat_memory.add_user_message(user_input)
    conversation_memory.chat_memory.add_ai_message(response.content)
    
    # Log the AI's response to the console
    print(f"AI: {response.content}")
    
    return jsonify({
        "response": response.content,
        "history": serialize_messages(conversation_memory.chat_memory.messages)
    })

if __name__ == "__main__":
    app.run(debug=True)