from langchain.tools import tool

@tool
def greeting(name: str) -> str:
    """Generate the greeting message for a given name."""
    return f"Hello, {name}! Welcome to the AI world."

result = greeting.invoke("Mukaram")
print(result)

print(greeting.name)
print(greeting.description)
print(greeting.args)