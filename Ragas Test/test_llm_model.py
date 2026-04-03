"""
"RAGAS can be used beyond RAG pipelines to evaluate LLM outputs by treating ground truth as pseudo-context
and comparing model responses using semantic and relevance metrics."
"""
import os
from dotenv import load_dotenv
import pytest
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from transformers import pipeline
from datasets import Dataset as HFDataset
from ragas import evaluate, RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextEntityRecall,
    AnswerSimilarity
)
# val = "qa"
# load_dotenv(dotenv_path=f"{val}.env")
load_dotenv()

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key=os.getenv("GENAI_PLUS"),
    convert_system_message_to_human=True,
    n=1
)

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GENAI_PLUS"),
)

model_a = pipeline("text-generation", model="microsoft/phi-2")
model_b = pipeline("text-generation", model="tiiuae/falcon-7b-instruct")

data = [
    {
        "question": "What is Python?",
        "ground_truth": "Python is a programming language."
    },
    {
        "question": "What is API?",
        "ground_truth": "API allows communication between systems."
    }
]

def eval_dataset_creation():
    eval_data = []

    for item in data:
        q = item["question"]

        answer_a = model_a(q)[0]["generated_text"]
        answer_b = model_b(q)[0]["generated_text"]

        eval_data.append({
            "question": q,
            "ground_truth": item["ground_truth"],
            "answer_model_a": answer_a,
            "answer_model_b": answer_b,
        })
    return eval_data




@pytest.mark.parametrize("model", ("model_a", "model_b"), ids=("model_a", "model_b"))
def test_ragas_val_per_model(model):
    ragas_data = []
    eval_data = eval_dataset_creation()
    for row in eval_data:
        ragas_data.append({
            "question": row["question"],
            "ground_truth": row["ground_truth"],
            "response": row[f"answer_{model}"],  # RAGAS expects 'response' column
            "retrieved_contexts": [row["ground_truth"]]  # RAGAS expects 'retrieved_contexts' column
        })
    # Create dataset from ragas_data (with correct column names)
    dataset = HFDataset.from_list(ragas_data)
    print(f"Dataset ready for model: {model}")
    result = evaluate(
        dataset,
        metrics=[
            Faithfulness(),
            AnswerRelevancy(),
            ContextPrecision(),
            ContextRecall(),
            # ContextEntityRecall(),
        ],
        llm=llm,
        embeddings=embedding_model,
        run_config=RunConfig(
            timeout=120,  # Sets a timeout for each evaluation task (per metric/sample)
            max_workers=1  # Controls parallelization -limits to 1 concurrent worker thread
        )
    )
    print(f"Evaluation complete for model: {model}")
    print(result)

