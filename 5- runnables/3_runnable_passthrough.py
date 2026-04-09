from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnablePassthrough

load_dotenv()

code_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a expert python code generator."),
    ("human", "{topic}")
])

explain_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an expert python code explainer. Explain the complete code line by line in simple words."),
    ("human", "{code}")
])

parser = StrOutputParser()

model = ChatGroq(model="groq/compound-mini", temperature=0.7)

seq_1 = code_prompt | model | parser

seq_2 = RunnableParallel({
    "code": RunnablePassthrough(),
    "explanation": explain_prompt | model | parser
})

chain = seq_1 | seq_2

response = chain.invoke("Write a python function to calculate the factorial of a number.")

print(response)
print("")
print(response["code"])
print("Explanation:")
print(response["explanation"])

# Runnable Passthrough returns the input as output without any modification. It is useful when you want to pass the output of one component to another component without any changes.