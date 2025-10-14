from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import torch.nn.functional as F

# Load model and tokenizer
model_path = "trained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Label mapping
id2label = {0: 'Positive', 1: 'Negative', 2: 'Risky'}

# Keyword helpers
risky_keywords = [
    "suicide", "kill myself", "end my life", "depressed", "hopeless",
    "worthless", "anxious", "panic", "cut myself", "die", "alone", "crying"
]

neutral_keywords = [
    "no problem", "everything is fine", "i am okay", "all good",
    "not bad", "feeling fine", "doing okay","dont have any problem"
]

def predict_text(text):
    text_lower = text.lower().strip()

    # Handle empty or too short input
    if not text_lower or len(text_lower.split()) < 2:
        return "Positive", 1.0, None  # Treat as neutral positive

    # Model prediction
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    probs = F.softmax(outputs.logits, dim=-1)
    conf, pred_id = torch.max(probs, dim=1)

    label = id2label[pred_id.item()]
    confidence = conf.item()

    # Rule-based overrides
    if any(word in text_lower for word in risky_keywords):
        label = "Risky"
        confidence = max(confidence, 0.95)

    elif any(word in text_lower for word in neutral_keywords):
        label = "Positive"
        confidence = max(confidence, 0.95)

    return label, confidence, probs.detach().numpy().flatten()


if __name__ == "__main__":
    # Example manual tests
    while True:
        user_text = input("Enter text (or 'exit'): ")
        if user_text.lower() == "exit":
            break
        label, conf, _ = predict_text(user_text)
        print(f"{user_text} --> {label} ({conf:.2f})")
