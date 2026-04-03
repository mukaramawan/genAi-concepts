""" There are 2 ways to use open source models from Hugging Face.
    1. Through API
    2. By running them locally
"""

from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint

load_dotenv()

llm = HuggingFaceEndpoint(
    repo_id="deepseek-ai/DeepSeek-R1",
)

model = ChatHuggingFace(llm=llm)

response = model.invoke("What is Cricket?")
print(response.content)