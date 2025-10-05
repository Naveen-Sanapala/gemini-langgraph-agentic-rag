from langchain_google_genai import GoogleGenerativeAIEmbeddings
import numpy as np
import os
from dotenv import load_dotenv
# Load variables from .env into environment
load_dotenv()

google_api_key = os.getenv("GOOGLE_API_KEY")
class Embedder:
    def __init__(self):
        self.model= GoogleGenerativeAIEmbeddings(
        google_api_key=google_api_key,
        model ="models/text-embedding-004"
        )

    def embed_texts(self,texts):
        embs =self.model.embed_documents(texts)
        #print(len(embs))
        return np.array(embs,dtype=np.float32)
     
# a=Embedder()
# b=a.embed_texts(["What is the capital of india"])
# print(b[0])