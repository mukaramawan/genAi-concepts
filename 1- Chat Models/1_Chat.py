from dotenv import load_dotenv
load_dotenv()

from langchain.chat_models import init_chat_model

model = init_chat_model('groq:groq/compound-mini')

response = model.invoke("What is Cricket?")

print(response.content)