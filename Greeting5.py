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
behavioral_llm = ChatMistralAI(model="mistral-large-latest", temperature=0.6)  # For behavioral questions
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

behavioral_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are conducting a behavioral interview. Ask questions that reveal 
    the candidate's soft skills, problem-solving approach, and cultural fit.
    Ask follow-up questions when interesting points emerge.
    
    Conversation history:
    {history}
    
    Candidate: {input}
    Interviewer:"""
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

behavioral_chain = ConversationChain(
    llm=behavioral_llm,
    memory=memory,
    prompt=behavioral_prompt
)

technical_chain = ConversationChain(
    llm=technical_llm,
    memory=memory,
    prompt=technical_prompt
)

# --- Interview Flow ---
def run_interview():
    tech_questions = load_technical_questions("C:\\Users\\Ezzwa\\Desktop\\New folder (3)\\Software Questions.csv")
    behavioral_questions = [
        "Tell me about a time you faced a difficult challenge at work and how you handled it",
        "Describe a situation where you had to work with a difficult team member",
        "Give an example of when you took initiative to improve a process"
    ]
    
    print("Starting interview...\n")

    # 1. Greeting Phase (4 exchanges)
    print("(Greeting Phase)")
    response = greeting_chain.predict(input="Greet the candidate warmly and wait for their response")
    print(f"Interviewer: {response}")
    user_input = input("Candidate: ")
    # 2. Small Talk (3 exchanges)
    for _ in range(3):
        response = greeting_chain.predict(input=user_input)
        print(f"\nInterviewer: {response}")
        user_input = input("Candidate: ")

    # 3. Behavioral Questions (2-3 questions with follow-ups)
    print("\n(Behavioral Questions Phase)")
    selected_behavioral = random.sample(behavioral_questions, min(3, len(behavioral_questions)))
    
    for question in selected_behavioral:
        # Ask main behavioral question
        response = behavioral_chain.predict(input=f"""Comment briefly on the last thing they said and then ask this behavioral question: {question}
        Do not mention anything about the interview process. Do not ask multiple questions at once. Do not say 'thank you for your answer'.
        """)
        print(f"\nInterviewer: {response}")
        candidate_answer = input("Candidate: ")
        
        # Generate follow-up based on answer
        follow_up = behavioral_chain.predict(
            input=f"""Candidate answered: {candidate_answer}
            Generate one relevant follow-up question to dig deeper"""
        )
        print(f"\nInterviewer: {follow_up}")
        candidate_followup = input("Candidate: ")
        
        # Optional: Second follow-up if interesting
        if "interesting" in behavioral_chain.predict(input=f"Was this answer interesting? {candidate_followup}").lower():
            second_followup = behavioral_chain.predict(
                input=f"Ask a second follow-up based on: {candidate_followup}"
            )
            print(f"\nInterviewer: {second_followup}")
            input("Candidate: ")

    # 4. Technical Questions (3 questions with evaluation)
    print("\n(Technical Questions Phase)\n")
    selected_technical = random.sample(tech_questions, 3)  # Get 3 unique random questions
    
    for question_data in selected_technical:
        question = question_data['Question']
        model_answer = question_data['Answer']
        
        # Ask question
        response = technical_chain.predict(input=f"Ask this technical question: {question}")
        print(f"\nInterviewer: {response}")
        candidate_answer = input("Candidate: ")
        
        # Evaluate answer
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