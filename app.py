import streamlit as st
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F
import time
from huggingface_hub import login
import os

# --------------------------
# Load model and tokenizer
# --------------------------
hf_token = st.secrets["HUGGINGFACE_TOKEN"]
login(hf_token)
MODEL_PATH = "RohitSK146/mental_health_companion"
tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH,use_auth_token=hf_token)
model = AutoModelForSequenceClassification.from_pretrained(MODEL_PATH,use_auth_token=hf_token)

# --------------------------
# Label Mapping
# --------------------------
id2label = {0: 'Positive', 1: 'Negative', 2: 'Risky'}

# --------------------------
# Risky Keywords
# --------------------------
risky_keywords = [
    "suicide", "kill myself", "end my life", "depressed", "hopeless", "worthless",
    "anxious", "panic", "cut myself", "die", "alone", "crying", "empty",
    "tired of life", "no reason to live", "done with life", "lost all hope"
]

neutral_keywords = [
    "no problem", "everything is fine", "i am okay", "all good", "not bad", "feeling fine","dont have any problem"
]



# --------------------------
# Helper: Predict Function
# --------------------------
def predict_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)

    probs = F.softmax(outputs.logits, dim=-1)
    conf, pred_id = torch.max(probs, dim=1)

    label = id2label[pred_id.item()]
    confidence = conf.item()

    




# Inside predict_text():
    text_lower = text.lower()

# Rule-based Risky detection
    if any(word in text_lower for word in risky_keywords):
      if confidence < 0.9 or label == "Negative":
        label = "Risky"

# Rule-based Neutral/Positive correction
    elif any(word in text_lower for word in neutral_keywords):
        label = "Positive"
        confidence = max(confidence, 0.95)  # Give high confidence to override


    return label, confidence, probs.detach().numpy().flatten()

# --------------------------
# UI Configuration
# --------------------------
st.set_page_config(page_title="🧠 Mental Health Companion", page_icon="💬", layout="centered")

st.title("🧠 Mental Health Companion")
st.write("Your emotional insight assistant — analyze your thoughts and feelings instantly.")

# --------------------------
# History Storage
# --------------------------
if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------
# User Input
# --------------------------
user_input = st.text_area("💬 Share what’s on your mind:", "", height=120)

if st.button("✨ Analyze My Mood"):
    if user_input.strip():
        with st.spinner("Analyzing your mood..."):
            time.sleep(1)
            label, conf, probs = predict_text(user_input)

        # Save to history
        st.session_state.history.insert(0, {"text": user_input, "label": label, "confidence": conf})

        # --------------------------
        # Emoji and Message Mapping
        # --------------------------
        messages = {
            "Positive": ("🌞", "You’re radiating good energy! Keep nurturing that positivity."),
            "Negative": ("🌧️", "It seems you’re feeling low. Remember, dark clouds pass too."),
            "Risky": (
        "🚨",
        """You might be going through distress. Please don’t face it alone — reach out to someone you trust or contact a helpline immediately.

📞 **India Helpline Numbers:**
- AASRA Helpline: 91-9820466726  
- Vandrevala Foundation Helpline: 1860 266 2345 / 9999 666 555  
- iCall: +91 9152987821  
- Snehi Helpline: +91-9582208181  

💬 If you are outside India, please look for your local mental health helpline or visit [findahelpline.com](https://findahelpline.com).

"""
    )
        }

        emoji, msg = messages[label]
        st.markdown(f"### {emoji} **{label}** ({conf*100:.1f}% confidence)")
        st.markdown(f"_{msg}_")

        # --------------------------
        # Confidence Bar Chart
        # --------------------------
        st.write("### 🔍 Emotion Confidence Breakdown:")
        st.bar_chart({
            "Positive": [probs[0]],
            "Negative": [probs[1]],
            "Risky": [probs[2]],
        })

    else:
        st.warning("Please enter something to analyze!")

# --------------------------
# History Section
# --------------------------
if st.session_state.history:
    st.markdown("## 🕒 Recent Mood Checks")
    for i, entry in enumerate(st.session_state.history[:5]):
        emoji = "🌞" if entry["label"] == "Positive" else ("🌧️" if entry["label"] == "Negative" else "🚨")
        st.markdown(f"**{emoji} {entry['label']}** — *{entry['text']}* ({entry['confidence']*100:.1f}%)")
