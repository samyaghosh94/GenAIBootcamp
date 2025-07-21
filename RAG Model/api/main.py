from fastapi import FastAPI
from pydantic import BaseModel
from rag_pipeline.rag import gemini_rag_pipeline

app = FastAPI()

class QueryRequest(BaseModel):
    base_url: str
    question: str
    max_links: int = 10

@app.post("/ask")
def ask_question(req: QueryRequest):
    answer = gemini_rag_pipeline(req.base_url, req.question, req.max_links)
    return {"answer": answer}
