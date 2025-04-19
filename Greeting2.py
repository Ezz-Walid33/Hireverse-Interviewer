from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
from langchain_mistralai.chat_models import ChatMistralAI
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
load_dotenv()
# Initialize LLM with persistent memory
model = ChatMistralAI(model="mistral-large-latest", temperature=0.7)
memory = ConversationBufferMemory()

# Custom prompt template that uses full history
interview_prompt = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are a professional interviewer conducting a job interview. 
    Maintain a friendly but professional tone. The conversation history is below:
    
    {history}
    
    Candidate: {input}
    Interviewer:"""
)

# Create master conversation chain
conversation = ConversationChain(
    llm=model,
    memory=memory,
    prompt=interview_prompt
)

# --- Interview Flow ---
print("Starting interview...\n")

# 1. Greeting
response = conversation.predict(input="Greet the candidate warmly")
print(f"Interviewer: {response}")

# 2. Small Talk (3 exchanges)
for i in range(3):
    user_input = input("Candidate: ")
    response = conversation.predict(input=user_input)
    print(f"\nInterviewer: {response}")

# 3. Transition to Behavioral Questions
behavioral_questions = [
    "Tell me about a time you faced a significant challenge at work",
    "Describe a situation where you handled conflict in a team",
    "Share an example of when you had to learn something new quickly"
]
print("\n(Transitioning to behavioral questions)\n")

for i, question in enumerate(behavioral_questions):
    # Ask question
    response = conversation.predict(input=f"Comment briefly on the last thing the candidate said and then Ask this behavioral question: {question}")
    print(f"\nInterviewer: {response}")
    
    # Get answer
    user_input = input("Candidate: ")
    conversation.predict(input=user_input)  # Store answer
    
    # Follow-up (context-aware)
    if i < len(behavioral_questions) - 1:  # Not for last question
        follow_up = conversation.predict(
            input="Ask one natural follow-up question based on their last answer"
        )
        print(f"\nInterviewer: {follow_up}")
        user_input = input("Candidate: ")
        conversation.predict(input=user_input)

# Show full history
print("\n--- Complete Transcript ---")
print(memory.load_memory_variables({})['history'])