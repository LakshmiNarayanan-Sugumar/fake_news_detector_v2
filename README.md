# Fake News Detector V2

## What is this?
A web app that detects whether a news article is real or fake using BERT.
Just paste any news text and it gives you a prediction with a confidence score.

## Why I built this?
This is the upgraded version of my V1 project, where I used TF-IDF + Logistic Regression.
I wanted to try a transformer-based approach and see how much better it performs on the same problem.

## A bug I found (and how I fixed it)
After training, my model showed 99.53% accuracy on the test set — but when I tried it on articles outside the WELFake dataset (like satire and hoax articles I wrote/collected myself), it did really badly, only around 53.8% accuracy. So the high accuracy number was misleading.

I dug into the training data to figure out why, and found that 61.8% of "real" articles in WELFake mentioned the word "Reuters," but only 1.8% of "fake" articles did. The model had basically learned "if it says Reuters, it's real" instead of actually learning what makes news fake or real. That's a classic shortcut-learning problem — it works great on the benchmark but doesn't generalize.

**What I did about it:**
- Built a small (26-example) out-of-distribution test set by hand, covering satire, hoaxes, and real articles from sources other than Reuters, to actually measure generalization instead of just trusting the benchmark accuracy.
- Cleaned the training data to remove the shortcut — stripped "Reuters" mentions, normalized ALL-CAPS text, and capped excessive exclamation marks.
- Retrained the model on this cleaned data.

**Result:** accuracy on real news from non-Reuters sources went from 87.5% to 100% on my OOD test set, so the shortcut is fixed. Satire and hoax detection are still weak (didn't really improve), which makes sense — WELFake just doesn't have much calm-toned satire or dry hoax writing in its "fake" class, so the model never learned to recognize that style as suspicious. That's a data problem, not something more cleaning can fix — next step would be adding a more diverse dataset like ISOT.

I think this was a more useful outcome than just reporting the accuracy number, since it shows the model actually generalizes better now, and I know exactly what its remaining weakness is instead of pretending it doesn't have one.

Full write-up with the before/after numbers is on the [model card](https://huggingface.co/LakshmiNarayanan-sugumar/fake-news-bert-debiased).

## Results (after fixing the shortcut)
- Accuracy: 99.49%
- F1 Score: 99.42%
- Precision: 99.64%
- Recall: 99.20%
- Dataset: WELFake, cleaned to 62,200 deduplicated articles
- Training: 2 epochs, fine-tuned `bert-base-uncased`

## Live Demo
https://huggingface.co/spaces/LakshmiNarayanan-sugumar/fake-news-detector

## Models
- [`fake-news-bert-debiased`](https://huggingface.co/LakshmiNarayanan-sugumar/fake-news-bert-debiased) — current version, use this one
- [`fake-news-bert`](https://huggingface.co/LakshmiNarayanan-sugumar/fake-news-bert) — older version with the shortcut bug, kept public so the fix is documented

## Files
- `app.py` — Streamlit app
- `ood_test.csv` — the 26-example out-of-distribution test set I built to catch this bug
- `requirements.txt` — dependencies

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