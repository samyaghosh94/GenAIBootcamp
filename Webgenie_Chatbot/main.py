import asyncio
import os
from dotenv import load_dotenv

from autogen_agentchat.teams import Swarm
from autogen_agentchat.ui import Console
from autogen_agentchat.conditions import HandoffTermination
from autogen_agentchat.messages import HandoffMessage
from autogen_agentchat.agents import AssistantAgent
from langchain_openai import AzureOpenAIEmbeddings
from autogen_ext.models.openai import OpenAIChatCompletionClient
from tools.rag_tool import rag_tool
from autogen_core.models import ModelInfo, ModelFamily
from constants import MASTER_PROMPT


load_dotenv()

gemini_client = OpenAIChatCompletionClient(
    model="gemini-1.5-flash",  # or "gemini-1.5-pro"
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
    api_key=os.getenv("GENAI_KEY"),
    model_info=ModelInfo(
        family="gemini",
        vision=False,
        function_calling=True,
        json_output=False
    )
)

# Create agents
router_agent = AssistantAgent(
    "router_agent",
    model_client=gemini_client,
    handoffs=["rag_agent", "user"],
    system_message="""You are a router agent. Your job is to receive user queries and forward them
                to the RAG agent.""",
)
# rag_agent = RagAgent(embeddings)

rag_agent = AssistantAgent(
    "rag_agent",
    model_client=gemini_client,
    tools=[rag_tool],
    handoffs=["router_agent","user"],
    system_message= MASTER_PROMPT
    # system_message=
    #             "You are a RAG agent assisting users with Ustora shopping queries. "
    #             "Use the 'retrieve_context' tool to get relevant information. "
    #             "Answer the user's question using only the retrieved context. Then STRICTLY handoff to the user.",
)

termination = HandoffTermination(target="user")
team = Swarm([router_agent, rag_agent], termination_condition=termination)

task = "How to add a new employee record?"


async def run_team_stream() -> None:
    """
    Runs the Swarm team in a loop, allowing for handoff-based interaction with the user.
    Ends when there are no further handoffs to the user, or when the user types 'exit'.
    """
    try:
        # Start the initial run
        task_result = await Console(team.run_stream(task=task))
        last_message = task_result.messages[-1]

        # Loop while the last message hands off to the user
        while isinstance(last_message, HandoffMessage) and last_message.target == "user":
            user_input = input("\nUser (type 'exit' to quit): ").strip()
            if user_input.lower() in {"exit", "quit"}:
                print("🛑 Exiting the chat.")
                break

            # Pass user input back into the team
            task_result = await Console(
                team.run_stream(
                    task=HandoffMessage(
                        source="user",
                        target=last_message.source,
                        content=user_input
                    )
                )
            )
            last_message = task_result.messages[-1]

    except Exception as e:
        print(f"❌ An error occurred during chat execution: {e}")


if __name__ == '__main__':
    asyncio.run(run_team_stream())
