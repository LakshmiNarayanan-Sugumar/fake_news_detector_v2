import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Page config
st.set_page_config(
    page_title="Fake News Detector V2",
    page_icon="🔍",
    layout="centered"
)

# Load model from HuggingFace Hub
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("LakshmiNarayanan-sugumar/fake-news-bert-debiased")
    model = AutoModelForSequenceClassification.from_pretrained("LakshmiNarayanan-sugumar/fake-news-bert-debiased")
    model.eval()
    return tokenizer, model

tokenizer, model = load_model()

# UI
st.title("Fake News Detector V2")
st.markdown("Powered by BERT fine-tuned on a cleaned WELFake dataset (62,200 articles) — 99.49% test accuracy")
st.markdown(
    "This version fixes a shortcut-learning bug found during evaluation (the model had learned to "
    "associate the word 'Reuters' with real news). Full diagnosis and before/after results are documented "
    "in the [model card](https://huggingface.co/LakshmiNarayanan-sugumar/fake-news-bert-debiased). "
    "**Known limitation:** the model still struggles with satire and dry-toned hoaxes — see model card for details."
)
st.markdown("---")

news_input = st.text_area("Paste a news article or headline below:", height=200, placeholder="Enter news text here...")

if st.button("Analyze", use_container_width=True):
    if not news_input.strip():
        st.warning("Please enter some text to analyze.")
    else:
        with st.spinner("Analyzing..."):
            inputs = tokenizer(
                news_input,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            with torch.no_grad():
                outputs = model(**inputs)

            probs = F.softmax(outputs.logits, dim=-1)
            pred = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][pred].item() * 100

            if pred == 1:
                st.error(f"FAKE NEWS — {confidence:.2f}% confidence")
            else:
                st.success(f"REAL NEWS — {confidence:.2f}% confidence")

            st.markdown("---")
            col1, col2 = st.columns(2)
            col1.metric("Real", f"{probs[0][0].item()*100:.2f}%")
            col2.metric("Fake", f"{probs[0][1].item()*100:.2f}%")