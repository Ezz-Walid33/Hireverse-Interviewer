from dotenv import load_dotenv
from langchain_mistralai.chat_models import ChatMistralAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

# Load environment variables from .env
load_dotenv()

# Create a ChatMistralAI model
model = ChatMistralAI(
    model="open-mistral-7b",
    temperature=0.7  # Slightly higher for more natural responses
)

# Improved system message
system_message = SystemMessage(content="""
You are conducting a technical interview in a natural, conversational style. Follow these guidelines:

1. Ask one behavioral question at a time
2. Listen carefully to responses and ask relevant follow-ups
3. Keep questions open-ended
4. Use natural language (e.g., "That's interesting, tell me more about...")
5. Maintain professional but friendly tone
6. Don't mention the interview process itself
7. Transition smoothly between questions

Behavioral questions to choose from (ask one at a time):
- Tell me about a challenging project you worked on
- Describe a time you had to learn something new quickly
- Share an example of resolving conflict in a team
- Explain a situation where you had to adapt to change
""")

chat_history = [system_message]

# Natural opening prompt
opening_prompt = HumanMessage(content="Hello, I'm ready to begin the interview.")
chat_history.append(opening_prompt)

# Get first question
ai_response = model.invoke(chat_history)
first_question = ai_response.content
chat_history.append(AIMessage(content=first_question))
print(f"Interviewer: {first_question}")

while True:
    user_input = input("You: ")
    if user_input.lower() in ["quit", "end", "exit"]:
        break
        
    chat_history.append(HumanMessage(content=user_input))
    
    # Get response with conversation history
    ai_response = model.invoke(chat_history)
    response = ai_response.content
    chat_history.append(AIMessage(content=response))
    
    print(f"\nInterviewer: {response}\n")

# Print full conversation history if needed
print("\n---- Interview Transcript ----")
for msg in chat_history:
    if isinstance(msg, HumanMessage):
        print(f"Candidate: {msg.content}")
    elif isinstance(msg, AIMessage):
        print(f"Interviewer: {msg.content}")