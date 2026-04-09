from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableParallel, RunnableLambda

load_dotenv()

short_prompt = ChatPromptTemplate.from_template(
    "Explain the {topic} in simple words of 2-3 sentences only, and give an example."
)

detail_prompt = ChatPromptTemplate.from_template(
    "Explain the {topic} in detail, and give an example."
)

parser = StrOutputParser()

model = ChatGroq(model="groq/compound-mini", temperature=0.7)

# chains = RunnableParallel({
#     "short_response":  short_prompt | model | parser,
#     "detailed_response": detail_prompt | model | parser
# })
# response = chains.invoke("What is deep Learning?")


# Parallel Runnables
# In a parallel runnable, you can define multiple branches of execution that run concurrently. Each branch can have its own prompt, model, and parser, allowing you to generate different types of responses for the same input. In the example below, we create a parallel runnable that generates both a short response and a detailed response for a given topic.

# The RunnableLambda is used to extract the relevant input for each branch from the overall input. It is used to add custom logic inside our runnable flow.

chains = RunnableParallel({
    "short_response": RunnableLambda(lambda x : x["short_response"]) | short_prompt | model | parser,
    "detailed_response": RunnableLambda(lambda x : x["detailed_response"]) | detail_prompt | model | parser
})


response = chains.invoke({
    "short_response": "What is Machine Learning?",
    "detailed_response": "What is Deep Learning?"
})

print(response["short_response"])
print("")
print(response["detailed_response"])

