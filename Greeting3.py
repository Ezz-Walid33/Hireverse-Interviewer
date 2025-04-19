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
    # Specify the encoding to handle non-UTF-8 characters
    df = pd.read_csv(csv_path, encoding="latin1")  # Try "cp1252" if "latin1" doesn't work
    return df[['Question', 'Answer']].to_dict('records')

# Initialize LLM with persistent memory
model = ChatMistralAI(model="mistral-large-latest", temperature=0.7)
memory = ConversationBufferMemory()

# Custom prompt template
interview_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are a professional interviewer. Maintain a friendly but professional tone.
    
    Conversation history:
    {history}
    
    Candidate: {input}
    Interviewer:"""
)

# Create conversation chain
conversation = ConversationChain(
    llm=model,
    memory=memory,
    prompt=interview_prompt
)

# --- Interview Flow ---
def run_interview():
    # Load questions
    tech_questions = load_technical_questions("C:\\Users\\Ezzwa\\Desktop\\New folder (3)\\Software Questions.csv")
    
    print("Starting interview...\n")

    # 1. Greeting
    response = conversation.predict(input="Greet the candidate warmly")
    print(f"Interviewer: {response}")

    # 2. Small Talk (3 exchanges)
    for _ in range(3):
        user_input = input("Candidate: ")
        response = conversation.predict(input=user_input)
        print(f"\nInterviewer: {response}")

    # 3. Technical Questions
    print("\n(Transitioning to technical questions)\n")
    
    for _ in range(3):  # Ask 3 technical questions
        # Select random question
        question_data = random.choice(tech_questions)
        question = question_data['Question']
        model_answer = question_data['Answer']
        
        # Ask question
        response = conversation.predict(input=f"Ask this technical question: {question}")
        print(f"\nInterviewer: {response}")
        
        # Get candidate's answer
        candidate_answer = input("Candidate: ")
        
        # Compare with model answer
        evaluation = conversation.predict(
            input=f"""Compare these answers for the question '{question}':
            Model Answer: {model_answer}
            Candidate Answer: {candidate_answer}
            Provide brief feedback (1-2 sentences) and a score from 1-10"""
        )
        print(f"\nEvaluation: {evaluation}")

    # Show full history
    print("\n--- Complete Transcript ---")
    print(memory.load_memory_variables({})['history'])

# Run interview
run_interview()