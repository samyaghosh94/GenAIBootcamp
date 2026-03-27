import os
import traceback
import asyncio
from pathlib import Path
import json

from dotenv import load_dotenv
from ragas import experiment
from rag_pipeline import rag_pipeline
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()

# Initialize models
# llm = ChatGoogleGenerativeAI(
#     model="gemini-1.5-pro",  # Use a stable model
#     temperature=0.0,
#     google_api_key=os.getenv("GENAI_PLUS"),
#     convert_system_message_to_human=True
# )

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/embedding-001",  # Use stable embedding model
    google_api_key=os.getenv("GENAI_PLUS"),
)

# Define the experiment without metrics in decorator
@experiment()
async def rag_experiment(row, temperature: float = 0.0, top_k: int = 3):
    """
    RAGAS experiment function that processes a single question row.
    
    Args:
        row: Dataset row containing question and ground_truth
        temperature: LLM temperature for generation
        top_k: Number of documents to retrieve
    
    Returns:
        Dict with question, answer, contexts, and ground_truth
    """
    query = row["question"]
    print(f"🔍 Processing: {query[:50]}...")

    # Get RAG response
    response, docs = rag_pipeline(
        query,
        temperature=temperature,
        top_k=top_k
    )

    # Return in RAGAS expected format
    return {
        "question": query,
        "ground_truth": row["ground_truth"],
        "answer": response,
        "contexts": [doc.page_content for doc in docs],
    }


def load_eval_dataset(questions_file):
    """
    Load evaluation dataset from JSON file.
    
    Expected format: [{"question": "...", "ground_truth": "..."}, ...]
    """
    if not Path(questions_file).exists():
        raise FileNotFoundError(f"Questions file not found: {questions_file}")

    with open(questions_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    print(f"📋 Loaded {len(data)} questions from {questions_file}")
    return data


async def run_rag_experiment(questions_file):
    """
    Run the RAGAS experiment on the dataset.
    """
    try:
        print("🚀 Starting RAGAS Experiment")
        print("=" * 50)

        # Load the raw data
        data = load_eval_dataset(questions_file)

        print(f"📊 Running experiment on {len(data)} samples...")

        results = []
        for i, row in enumerate(data):
            print(f"\n🔍 Processing sample {i+1}/{len(data)}: {row['question'][:50]}...")
            
            # Run the experiment on this single row
            result = await rag_experiment(
                row,
                temperature=0.0,
                top_k=2,
                llm=llm,
                embeddings=embedding_model
            )
            results.append(result)

        print("\n📈 Experiment Results:")
        print("=" * 50)
        for i, result in enumerate(results):
            print(f"Sample {i+1}:")
            print(f"  Question: {result['question'][:60]}...")
            print(f"  Answer: {result['answer'][:60]}...")
            print(f"  Contexts: {len(result['contexts'])} retrieved")
            print()

        return results

    except Exception as e:
        print(f"❌ Experiment failed: {e}")
        traceback.print_exc()
        return None


if __name__ == '__main__':
    # Use the correct questions file path
    questions_file = "./Data/questions.json"

    # Run the experiment
    results = asyncio.run(run_rag_experiment(questions_file))

    if results:
        print("\n✅ Experiment completed successfully!")
    else:
        print("\n❌ Experiment failed!")
