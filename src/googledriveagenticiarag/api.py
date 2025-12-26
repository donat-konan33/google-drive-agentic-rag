
import sys, os
sys.path.append(os.path.dirname(__file__))

from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from googledriveagenticiarag.get_answers import rag_answer


app = FastAPI(title="RAG API")

class Question(BaseModel):
    question: str

@app.post("/rag")
async def query_rag(payload: Question):
    return rag_answer(payload.question)

if __name__ == "__main__":
    uvicorn.run("googledriveagenticiarag.api:app", host="0.0.0.0", port=8001, reload=True)
