"""
API FastAPI exposant l'agent d'analyse financière.

Usage:
    uvicorn src.api.main:app --reload
"""

from fastapi import FastAPI
from pydantic import BaseModel

from src.rag.agent import answer

app = FastAPI(title="FinAgent API", version="0.1.0")


class Question(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ask")
def ask(payload: Question):
    return {"question": payload.question, "answer": answer(payload.question)}
