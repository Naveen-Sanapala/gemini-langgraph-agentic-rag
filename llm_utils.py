from langchain_google_genai import ChatGoogleGenerativeAI
import os
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()

# Access them
google_api_key = os.getenv("GOOGLE_API_KEY")
def llm_load():
    # Initialize chat LLM using API key
    try:
        llm = ChatGoogleGenerativeAI(
            google_api_key=google_api_key,
            model="gemini-2.5-flash-lite",  # Chat-focused model
            temperature=0.7
        )
    except Exception as e:
        print(e)
        llm=None
    return llm
llm=llm_load()

#print(llm.invoke("what is the capital of India"))
