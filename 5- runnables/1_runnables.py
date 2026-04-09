from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

prompt = ChatPromptTemplate.from_template(
    "Explain {topic} in simple words, and give an example."
)

parser = StrOutputParser()

model = ChatGroq(model="groq/compound-mini", temperature=0.7)

# formatted_prompt = prompt.format(topic="quantum computing")

# response = model.invoke(formatted_prompt)

# final_response = parser.parse(response.content)

# print(final_response)

# Runnables allow you to chain together multiple components, such as prompts, models, and parsers, in a single pipeline. This makes it easier to manage complex interactions and ensures that the output of one component can be seamlessly passed to the next. In the example below, we create a chain that takes a topic as input, generates a response using the model, and then parses the output to produce a final response.

# Sequential Runnables
chains = prompt | model | parser
response = chains.invoke("quantum computing")
print(response)

