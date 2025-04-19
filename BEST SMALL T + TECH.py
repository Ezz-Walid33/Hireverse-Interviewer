import pandas as pd
import random
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()

# Load technical questions from CSV
def load_technical_questions(csv_path):
    df = pd.read_csv(csv_path, encoding="latin1")
    return df[['Question', 'Answer']].to_dict('records')

# Initialize different LLMs for different phases
greeting_llm = ChatMistralAI(model="mistral-large-latest", temperature=0.7)  # Friendly and warm
technical_llm = ChatMistralAI(model="mistral-large-latest", temperature=0.5)  # Precise for technical evaluation

# Shared memory across all chains
memory = ConversationBufferMemory()

# Different prompt templates for each phase
greeting_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are a professional interviewer conducting a job interview. 
    Maintain a friendly but professional tone. The conversation history is below:
    
    {history}
    
    Human: {input}
    AI:"""
)

technical_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are a technical interviewer. Ask questions precisely and 
    evaluate answers professionally.
    
    Conversation history:
    {history}
    
    Candidate: {input}
    Interviewer:"""
)

# Create separate chains for each phase
greeting_chain = ConversationChain(
    llm=greeting_llm,
    memory=memory,
    prompt=greeting_prompt
)

technical_chain = ConversationChain(
    llm=technical_llm,
    memory=memory,
    prompt=technical_prompt
)

# --- Interview Flow ---
def run_interview():
    tech_questions = load_technical_questions("C:\\Users\\Ezzwa\\Desktop\\New folder (3)\\Software Questions.csv")
    
    print("Starting interview...\n")

    # 1. Greeting Phase (4 exchanges using greeting_llm)
    print("(Greeting Phase)")
    response = greeting_chain.predict(input="Greet the candidate warmly and wait for their response")
    print(f"Interviewer: {response}")

    # 2. Small Talk (3 exchanges)
    for _ in range(3):
        user_input = input("Candidate: ")
        response = greeting_chain.predict(input=user_input)
        print(f"\nInterviewer: {response}")

    # 2. Technical Questions (3 questions with evaluation)
    print("\n(Technical Questions Phase)\n")
    
    selected_questions = random.sample(tech_questions, 3)  # Get 3 unique random questions
    
    for question_data in selected_questions:
        question = question_data['Question']
        model_answer = question_data['Answer']
        
        # Ask question using technical chain
        response = technical_chain.predict(input=f"Ask this technical question: {question}")
        print(f"\nInterviewer: {response}")
        
        # Get candidate's answer
        candidate_answer = input("Candidate: ")
        
        # Evaluate using technical LLM
        evaluation = technical_chain.predict(
            input=f"""Evaluate this answer for '{question}':
            Expected Key Points: {model_answer}
            Candidate Answer: {candidate_answer}
            Provide specific technical feedback (what was good/missing) 
            and a score from 1-10 with justification"""
        )
        print(f"\nEvaluation: {evaluation}")

    # Closing
    print("\nInterviewer: Thank you for your time today. We'll be in touch soon!")
    
    # Show full history
    print("\n--- Complete Transcript ---")
    print(memory.load_memory_variables({})['history'])

# Run interview
run_interview()