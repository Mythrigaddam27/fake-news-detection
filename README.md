# Fake News Detection – Hackathon Demo

**Theme:** Generative AI & LLM Applications (but using classical ML for explainability & speed)

## What it does

Paste any news article → get:

- REAL or FAKE verdict
- Confidence score
- Top keywords that influenced the decision (based on TF-IDF weights)

## Tech stack

- Backend: Flask
- ML: scikit-learn (TF-IDF + Logistic Regression)
- Frontend: pure HTML + CSS + vanilla JavaScript
- No deep learning, no external APIs, no npm build step required

## How to run (two options)

### Option 1 – Simplest (recommended for demo)

1. Run model training once  
   ```bash
   python model_training.py