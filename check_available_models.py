from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
import google.generativeai as genai
import os
from dotenv import load_dotenv

# Load variables from .env into environment
load_dotenv()
# ensure API key is set
api_key = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=api_key)

# list models
models = genai.list_models()
for m in models:
    if "embedding" in m.name.lower():  
        print(m.name )