"""
Serves the fine-tuned financial-sentiment classifier via REST.
Run: uvicorn app.main:app --reload
"""
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline

app = FastAPI(title="Financial Sentiment Classifier API", version="1.0.0")

MODEL_DIR = "model/finetuned-sentiment"
classifier = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR, top_k=None)


class TextRequest(BaseModel):
    text: str


class SentimentResult(BaseModel):
    label: str
    score: float


@app.post("/predict", response_model=list[SentimentResult])
async def predict(req: TextRequest):
    results = classifier(req.text)[0]
    results.sort(key=lambda r: r["score"], reverse=True)
    return [SentimentResult(label=r["label"], score=round(r["score"], 4)) for r in results]


@app.get("/health")
async def health():
    return {"status": "ok"}
