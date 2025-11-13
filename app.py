from flask import Flask, request, jsonify, render_template
from transformers import AutoModelForSequenceClassification, AutoTokenizer
import torch

app = Flask(__name__)

# Load models once
model1_name = "cardiffnlp/twitter-roberta-base-sentiment"
model2_name = "nlptown/bert-base-multilingual-uncased-sentiment"

model1 = AutoModelForSequenceClassification.from_pretrained(model1_name)
model2 = AutoModelForSequenceClassification.from_pretrained(model2_name)
tokenizer1 = AutoTokenizer.from_pretrained(model1_name)
tokenizer2 = AutoTokenizer.from_pretrained(model2_name)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text")

    # Tokenize inputs
    inputs1 = tokenizer1(text, return_tensors="pt", truncation=True, padding=True, max_length=512)
    inputs2 = tokenizer2(text, return_tensors="pt", truncation=True, padding=True, max_length=512)

    # Get logits
    outputs1 = model1(**inputs1).logits
    outputs2 = model2(**inputs2).logits

    # Convert to probabilities
    probs1 = torch.softmax(outputs1, dim=-1)
    probs2 = torch.softmax(outputs2, dim=-1)

    # Map 5→3 class for model2
    neg = probs2[:, 0] + probs2[:, 1]
    neu = probs2[:, 2]
    pos = probs2[:, 3] + probs2[:, 4]
    probs2_mapped = torch.stack([neg, neu, pos], dim=1)

    # Average predictions
    combined_probs = (probs1 + probs2_mapped) / 2
    combined_probs = torch.pow(combined_probs, 1.2)  # amplify middle confidence
    combined_probs = combined_probs / combined_probs.sum()

    labels = ["negative", "neutral", "positive"]

    idx = torch.argmax(combined_probs, dim=-1).item()
    sentiment = labels[idx]
    confidence = round(combined_probs[0][idx].item(), 3)

    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence
    })

if __name__ == "__main__":
    app.run(debug=True)
