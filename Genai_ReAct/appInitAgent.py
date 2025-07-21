import os
from langchain_openai.chat_models import AzureChatOpenAI
from langchain.agents import initialize_agent, Tool, AgentType

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Initialize the LLM
llm = AzureChatOpenAI(
    azure_deployment="gpt-4o-mini-2024-07-18",
    api_version=os.getenv("AZURE_API_VERSION"),
    api_key=os.getenv("DIAL_LAB_KEY"),
    azure_endpoint=os.getenv("AZURE_ENDPOINT")
)

# Define a simple calculator tool
def calculator_tool(input: str) -> str:
    try:
        return str(eval(input))
    except Exception as e:
        return f"Error: {str(e)}"

tools = [
    Tool(
        name="Calculator",
        func=calculator_tool,
        description="Performs basic arithmetic calculations."
    )
]

# Initialize the agent
agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True
)

# Run the agent with a sample query
response = agent.run("What is 12 * 8 + 5?")
print(response)

response = agent.run("What is the capital of France?")
print(response)
