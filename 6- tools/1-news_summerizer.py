# Example of prebuilt tools

from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_tavily import TavilySearch

load_dotenv()

news_prompt = ChatPromptTemplate.from_template(
    """You are a helpful assistant that summerizes the news articles in simple words of 2-3 sentences only for every article.
    {articles}
"""
)

parser = StrOutputParser()
model = ChatGroq(model="groq/compound-mini", temperature=0.7)
chain = news_prompt | model | parser

search_tool = TavilySearch(max_results=5)

news_results = search_tool.run("Latest news of US, China, Iran & Pakistan.")
print(news_results)
print("")
response = chain.invoke({"articles": news_results})
print(response)