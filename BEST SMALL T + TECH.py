from langchain_groq import ChatGroq
from dotenv import load_dotenv
load_dotenv()
greeting_llm = ChatGroq(model_name="deepseek-r1-distill-llama-70b", temperature=0.7)
msg = greeting_llm.invoke("Hello, world!") 
print(msg.content)  
