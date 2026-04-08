from dotenv import load_dotenv
load_dotenv()

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel
from typing import List, Optional
from langchain_core.output_parsers import PydanticOutputParser

class MovieInfo(BaseModel):
    movie_name: str
    director: Optional[str]
    release_year: Optional[int]
    genre: List[str]
    main_cast: List[str]
    setting_location: Optional[str]
    notable_highlights: Optional[str]
    imdb_rating: Optional[float]
    summary: str

parser = PydanticOutputParser(pydantic_object=MovieInfo)

model = ChatGroq(model="openai/gpt-oss-20b")

# Instantiation using from_template (recommended)
template = ChatPromptTemplate.from_messages(
    [
        ("system", """
You are an expert data extraction assistant. Your task is to analyze the provided paragraph about a movie and extract key information into a structured format.
         
Your task is to extract the following information from the text provided below. 

Paragraph to analyze:
"{text}"

Please provide the information in this exact format:

{format_instructions}

---
**Quick Summary**:
[Provide a 2-sentence overview of the paragraph's main points]
         """),
        ("human", """Extract the information from the following paragraph: {text}"""),
    ]
)

text = input("Enter a paragraph about a movie: ")
final_prompt = template.invoke({"text": text, "format_instructions": parser.get_format_instructions()})


response = model.invoke(final_prompt)
print(response.content)

print(parser.parse(response.content))


#Interstellar is a visually stunning science fiction epic directed by Christopher Nolan. Released in 2014, the film stars Matthew McConaughey, Anne Hathaway, Jessica Chastain, and Michael Caine. The story revolves around a group of astronauts who travel through a wormhole near Saturn in search of a new home for humanity as Earth faces environmental collapse. The movie was widely appreciated for its emotional depth, scientific accuracy, and Hans Zimmer’s powerful soundtrack. It holds a rating of 8.6 on IMDb and is often considered one of the greatest sci-fi films of the 21st century.