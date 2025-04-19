import pandas as pd
import random
import os
from langchain.memory import ConversationBufferMemory
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_community.document_loaders import PyPDFLoader
from dotenv import load_dotenv
import PyPDF2

load_dotenv()

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

def run_interview():
    # Load CV
    cv_path = os.path.join(os.path.expanduser("~"), "Desktop", "candidate_cv.pdf")
    cv_text = extract_cv_text(cv_path)
    print(f"CV Text: {cv_text}\n")
    
    tech_questions = load_technical_questions("C:\\Users\\Ezzwa\\Desktop\\New folder (3)\\Software Questions.csv")
    behavioral_questions = [
        "Tell me about a time you faced a difficult challenge at work and how you handled it",
        "Describe a situation where you had to work with a difficult team member",
        "Give an example of when you took initiative to improve a process"
    ]
    
    memory = ConversationBufferMemory(return_messages=True)
    
    print("Starting interview...\n")

    # 1. Greeting Phase
    print("(Greeting Phase)")
    greeting_chain = get_greeting_prompt(cv_text) | greeting_llm
    response = greeting_chain.invoke({
        "input": "Greet the candidate warmly.",
        "history": []
    })
    print(f"Interviewer: {response.content}")
    memory.chat_memory.add_user_message("Greet the candidate warmly")
    memory.chat_memory.add_ai_message(response.content)
    
    # 2. Small Talk (3 exchanges)
    for _ in range(3):
        user_input = input("Candidate: ")
        memory.chat_memory.add_user_message(user_input)
        
        history = memory.chat_memory.messages
        response = greeting_chain.invoke({
            "input": user_input,
            "history": history
        })
        print(f"\nInterviewer: {response.content}")
        memory.chat_memory.add_ai_message(response.content)

    # 3. Behavioral Questions
    print("\n(Behavioral Questions Phase)")
    behavioral_chain = behavioral_prompt | behavioral_llm
    selected_behavioral = random.sample(behavioral_questions, min(3, len(behavioral_questions)))
    
    for question in selected_behavioral:
        history = memory.chat_memory.messages
        response = behavioral_chain.invoke({
            "input": f"""Comment briefly on the last thing they said and then ask: {question}
            Don't mention the interview process or thank them.""",
            "history": history
        })
        print(f"\nInterviewer: {response.content}")
        memory.chat_memory.add_ai_message(response.content)
        
        candidate_answer = input("Candidate: ")
        memory.chat_memory.add_user_message(candidate_answer)
        
        # Follow-up
        history = memory.chat_memory.messages
        follow_up = behavioral_chain.invoke({
            "input": f"Generate one relevant follow-up based on: {candidate_answer}",
            "history": history
        })
        print(f"\nInterviewer: {follow_up.content}")
        memory.chat_memory.add_ai_message(follow_up.content)
        
        candidate_followup = input("Candidate: ")
        memory.chat_memory.add_user_message(candidate_followup)

    # 4. Technical Questions
    print("\n(Technical Questions Phase)\n")
    technical_chain = technical_prompt | technical_llm
    selected_technical = random.sample(tech_questions, 3)
    
    for question_data in selected_technical:
        question = question_data['Question']
        model_answer = question_data['Answer']
        
        history = memory.chat_memory.messages
        response = technical_chain.invoke({
            "input": f"Ask this technical question: {question}",
            "history": history
        })
        print(f"\nInterviewer: {response.content}")
        memory.chat_memory.add_ai_message(response.content)
        
        candidate_answer = input("Candidate: ")
        memory.chat_memory.add_user_message(candidate_answer)
        
        evaluation = technical_chain.invoke({
            "input": f"""Evaluate this answer for '{question}':
            Expected: {model_answer}
            Candidate: {candidate_answer}
            Provide specific feedback and score from 1-10""",
            "history": history
        })
        print(f"\nEvaluation: {evaluation.content}")
        memory.chat_memory.add_ai_message(evaluation.content)

    # Closing
    print("\nInterviewer: Thank you for your time today. We'll be in touch soon!")
    
    # Show full history
    print("\n--- Complete Transcript ---")
    for msg in memory.chat_memory.messages:
        print(f"{msg.type}: {msg.content}")

if __name__ == "__main__":
    run_interview()