# FinSent: Fine-Tuned Transformer for Financial News Sentiment

Fine-tunes DistilBERT for 3-class sentiment classification (positive /
negative / neutral) on financial news headlines, then serves it via a
REST API. Demonstrates transfer learning / fine-tuning workflow the
skill most 2026 AI/ML job listings explicitly call out (PyTorch,
NLP, model evaluation, GenAI exposure) rather than just calling an
external LLM API.

## Why this project

Shows the full transformer fine-tuning lifecycle:
- Tokenization & dataset preparation with Hugging Face `datasets`
- Fine-tuning a pretrained transformer (`distilbert-base-uncased`) with the
  `Trainer` API
- Proper train/eval split with accuracy + macro-F1 tracked per epoch
- Model checkpointing and serving via FastAPI

## Architecture

```
data/generate_data.py → data/train.jsonl, data/test.jsonl
                                │
                                ▼
                    scripts/train.py
        (DistilBERT fine-tuning via HF Trainer, 4 epochs)
                                │
                                ▼
              model/finetuned-sentiment/ (saved checkpoint)
                                │
                                ▼
                      app/main.py (FastAPI)
```

## Setup

```bash
git clone https://github.com/<your-username>/finsent-classifier.git
cd finsent-classifier
pip install -r requirements.txt

python data/generate_data.py     # generate train/test splits
python scripts/train.py          # fine-tune DistilBERT (needs internet for base checkpoint)
uvicorn app.main:app --reload    # serve predictions
```

> Note: `data/generate_data.py` creates a small synthetic dataset (150
> labeled headlines) so the pipeline is runnable end-to-end without external
> downloads for the data step. For a resume-grade result, swap in the public
> `financial_phrasebank` or `twitter-financial-news-sentiment` datasets from
> Hugging Face Hub same JSONL schema (`text`, `label`) works directly.

## API Usage

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Apex Financial shares surged after strong earnings guidance."}'
```

**Response**
```json
[
  {"label": "positive", "score": 0.9123},
  {"label": "neutral", "score": 0.0611},
  {"label": "negative", "score": 0.0266}
]
```

## Project Structure

```
├── data/
│   ├── generate_data.py
│   ├── train.jsonl
│   └── test.jsonl
├── scripts/
│   └── train.py
├── app/
│   └── main.py
├── model/                  # created after training
├── requirements.txt
└── README.md
```

## Possible Extensions
- Swap DistilBERT for a domain-pretrained model (FinBERT) and compare
- Add confusion matrix + per-class error analysis notebook
- Track experiments with Weights & Biases
- Quantize model (ONNX/int8) for faster CPU inference

## License
MIT
