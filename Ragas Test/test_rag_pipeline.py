"""
Integration test for the RAG pipeline with Gemini
"""
import json
import os
import time
import traceback
from pathlib import Path
import asyncio

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Set tiktoken cache directory before importing anything that uses it
os.environ["TIKTOKEN_CACHE_DIR"] = os.path.join(os.getcwd(), ".tiktoken_cache")
os.makedirs(os.environ["TIKTOKEN_CACHE_DIR"], exist_ok=True)

from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from datasets import Dataset as HFDataset
from langchain_core.documents import Document
from rag_pipeline import rag_pipeline, embeddings
from ragas import evaluate, Dataset, RunConfig
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
    ContextEntityRecall
)

# Pre-cache tiktoken encoding to avoid network issues
# try:
#     import tiktoken
#     print("🔄 Pre-caching tiktoken encoding...")
#     encoding = tiktoken.get_encoding("cl100k_base")
#     print("✅ Tiktoken encoding cached successfully")
# except Exception as e:
#     print(f"⚠️  Tiktoken pre-caching issue (may retry): {e}")

load_dotenv()

QUESTIONS_FILE = "./Data/ragas_evaluation_dataset.json"

llm = ChatGoogleGenerativeAI(
    model="gemini-2.5-flash",
    temperature=0.0,
    google_api_key=os.getenv("GENAI_PLUS"),
    convert_system_message_to_human=True
)

# llm = ChatOpenAI(
#     model="gpt-3.5-turbo",
#     temperature=0.0,
#     api_key=os.getenv("OPENAI_API_KEY")
# )

embedding_model = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001",
    google_api_key=os.getenv("GENAI_PLUS"),
)

def test_vector_store_connection():
    """Test if vector store can be accessed"""
    try:
        from rag_pipeline import load_faiss_vectorstore, embeddings
        vectorstore = load_faiss_vectorstore(embeddings)
        print("✓ FAISS vector store connection successful")
        return True
    except Exception as e:
        print(f"✗ FAISS vector store connection failed: {e}")
        return False


def test_gemini_connection():
    """Test if Gemini API key is configured"""
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        api_key = os.getenv("GENAI_PLUS")
        
        if not api_key:
            print("✗ GOOGLE_API_KEY not found in .env")
            return False
        
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.5-flash",
            google_api_key=api_key
        )
        
        # Test with a simple message
        response = llm.invoke("Say 'RAG Pipeline Ready!'")
        print(f"✓ Gemini connection successful")
        print(f"  Response: {response.content}")
        return True
    except Exception as e:
        print(f"✗ Gemini connection failed: {e}")
        return False


def test_embeddings():
    """Test if embeddings work"""
    try:
        from rag_pipeline import embeddings
        test_text = "Hello, this is a test"
        embedding = embeddings.embed_query(test_text)
        print(f"✓ Gemini embeddings working (dimension: {len(embedding)})")
        return True
    except Exception as e:
        print(f"✗ Gemini embeddings failed: {e}")
        return False


async def run_rag_evaluation(questions_file):
    try:
        if not Path(questions_file).exists():
            print(f"❌ Questions file not found: {questions_file}")
            return None
            
        with open(questions_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        eval_data = []
        
        # Process each question
        for item in dataset:
            if isinstance(item, dict) and "question" in item:
                query = item["question"]
                print(f"\n📝 Processing query: '{query}'")
                
                # Get RAG response
                response, docs = rag_pipeline(query, temperature=0.0, top_k=3)
                contexts = [doc.page_content for doc in docs]

                # Debug checks
                print(f"➡️ Answer: {response[:100]}...")
                if not contexts:
                    print("⚠️ No contexts retrieved!")
                else:
                    print(f"📄 Contexts retrieved: {len(contexts)}")
                
                # Prepare evaluation data
                eval_data.append({
                    "question": query,
                    "ground_truth": item.get("ground_truth", ""),
                    "answer": response,
                    "contexts": contexts,
                })
                print(f"Eval result: {eval_data}")
                
                # Add small delay to avoid rate limits
                await asyncio.sleep(60)

        # Create Dataset for evaluation
        eval_dataset = HFDataset.from_list(eval_data)
        for item in eval_data:
            if not item["contexts"]:
                print("⚠️ Empty contexts found!")
        
        print(f"\n🔄 Running RAGAS evaluation on {len(eval_dataset)} samples...")
        
        # Run evaluation
        result = evaluate(
            eval_dataset,
            metrics=[
                Faithfulness(),
                AnswerRelevancy(),
                ContextPrecision(),
                ContextRecall(),
                # ContextEntityRecall(),
            ],
            llm=llm,  # ✅ REQUIRED - OpenAI LLM
            embeddings=embedding_model,
            # embeddings=embeddings,  # ✅ REQUIRED - OpenAI Embeddings from rag_pipeline
            run_config=RunConfig(
                timeout=120,  # seconds
                max_workers=1
            )
        )

        print("\n📊 Evaluation Results:")
        print(result)

        # Optional: detailed breakdown
        try:
            df = result.to_pandas()
            print("\n📈 Detailed Results:")
            print(df)
        except:
            pass

        # Save results
        with open("ragas_results.json", "w") as f:
            json.dump(eval_data, f, indent=2)

        print("\n✅ Results saved to ragas_results.json")
        
        return result

    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        traceback.print_exc()
        return None

# def test_rag_pipeline(questions_file):
#     """Test the complete RAG pipeline"""
#     try:
#         eval_data = []
#
#         # Create a sample query
#         # query = "What should I do if I get error adding employee?"
#         if not Path(questions_file).exists():
#             print(f"❌ Questions file not found: {questions_file}")
#         with open(questions_file, 'r', encoding='utf-8') as f:
#             data = json.load(f)
#         if isinstance(data, list):
#             for item in data:
#                 if isinstance(item, dict):
#                     query = item["question"]
#                     print(f"\n📝 Testing RAG pipeline with query: '{query}'")
#                     response, docs = rag_pipeline(query, temperature=0.0, top_k=3)
#                     eval_data.append({
#                         "question": query,
#                         "ground_truth": item["ground_truth"],
#                         "answer": response,
#                         "contexts": [doc.page_content for doc in docs],
#                         "reference_contexts": item["contexts"]
#                     })
#                     time.sleep(15)
#         print(f"  Response: {response[:100]}...")
#         print(f"  Retrieved documents: {len(docs)}")
#         print(f"Final Evaluation Dataset: {eval_data}")
#
#         eval_dataset = Dataset.from_list(eval_data)
#
#
#         print(f"✓ RAG pipeline executed successfully")
#         return True
#     except Exception as e:
#         print(f"✗ RAG pipeline test failed: {e}")
#         print("  Note: This may fail if no documents are indexed yet.")
#         print("  Run example_usage.py first to create the vector store.")
#         return False


def test_setup():
    """Run all tests"""
    print("=" * 60)
    print("RAG Pipeline Integration Tests")
    print("=" * 60)

    tests = [
        ("Vector Store", test_vector_store_connection),
        ("Gemini API", test_gemini_connection),
        ("Embeddings", test_embeddings),
    ]

    results = []
    for test_name, test_func in tests:
        print(f"\n▶ Testing {test_name}...")
        try:
            results.append(test_func())
        except Exception as e:
            print(f"✗ Unexpected error: {e}")
            results.append(False)

    print("\n" + "=" * 60)
    print(f"Summary: {sum(results)}/{len(results)} tests passed")
    print("=" * 60)

    return all(results[:3])  # First 3 tests are critical


if __name__ == "__main__":
    print(test_setup())
    print(asyncio.run(run_rag_evaluation("./Data/questions.json")))
