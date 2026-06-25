# Fake News Detector V2

## What is this?
A web app that detects whether a news article is real or fake using BERT.
Just paste any news text and it gives you a prediction with confidence score.

## Why I built this?
This is the upgraded version of my V1 project where I used TF-IDF + Logistic Regression.
Wanted to try a transformer-based approach and see how much better it performs.

## Results
- Accuracy: 99.55%
- F1 Score: 99.56%
- Dataset: WELFake (71,537 articles)
- Training: 3 epochs, fine-tuned bert-base-uncased

## Live Demo
https://huggingface.co/spaces/LakshmiNarayanan-sugumar/fake-news-detector

## Tech Used
- Python
- HuggingFace Transformers
- PyTorch
- Streamlit

## How to run
```bash
pip install -r requirements.txt
streamlit run app.py
```