from flask import Flask, request, jsonify, render_template
import requests
import os

app = Flask(__name__)

# Models to call on the Hugging Face Inference API
MODEL1 = "cardiffnlp/twitter-roberta-base-sentiment"
MODEL2 = "nlptown/bert-base-multilingual-uncased-sentiment"

HF_API_TOKEN = os.environ.get("HF_API_TOKEN")  # set this in Vercel (Project Settings → Environment Variables)
HF_API_URL = "https://api-inference.huggingface.co/models/"

HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

def call_hf_model(model_name, text):
    """Call HF inference API and return list of label/score dicts (or None on error)."""
    url = HF_API_URL + model_name
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    resp = requests.post(url, headers=HEADERS, json=payload, timeout=30)
    if resp.status_code != 200:
        # try to return error info
        app.logger.error("HF API error %s: %s", resp.status_code, resp.text)
        return None
    return resp.json()

def parse_model1_output(hf_resp):
    """
    cardiffnlp twitter-roberta-base-sentiment typically has 3 labels: 0=negative,1=neutral,2=positive
    HF returns list like [{"label":"LABEL_0","score":0.7}, ...] or human labels.
    We'll robustly map numeric index -> [neg,neu,pos] using any label text we find.
    """
    probs = [0.0, 0.0, 0.0]
    if not isinstance(hf_resp, list):
        return None
    # try to detect by label names or numeric suffix
    for item in hf_resp:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        # many HF sentiment models use LABEL_0/LABEL_1... or "negative"/"neutral"/"positive"
        if "neg" in label:
            probs[0] = score
        elif "neu" in label:
            probs[1] = score
        elif "pos" in label:
            probs[2] = score
        else:
            # try to extract digits from LABEL_X
            import re
            m = re.search(r"(\d+)", label)
            if m:
                idx = int(m.group(1))
                if 0 <= idx < 3:
                    probs[idx] = score
    # fallback: if probs all zero, try to assign by order (not ideal)
    if sum(probs) == 0 and len(hf_resp) >= 3:
        for i in range(3):
            probs[i] = float(hf_resp[i].get("score", 0.0))
    return probs

def parse_model2_output(hf_resp):
    """
    nlptown/bert-base-multilingual-uncased-sentiment returns 1-5 star labels like '1 star', '2 stars', ...
    Map 1-2 -> negative, 3 -> neutral, 4-5 -> positive.
    """
    neg, neu, pos = 0.0, 0.0, 0.0
    if not isinstance(hf_resp, list):
        return None
    for item in hf_resp:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        # try to find numeric star
        import re
        m = re.search(r"(\d)", label)
        if m:
            val = int(m.group(1))
            if val <= 2:
                neg += score
            elif val == 3:
                neu += score
            else:
                pos += score
        else:
            # fallback: look for words
            if "neg" in label:
                neg += score
            elif "neu" in label:
                neu += score
            elif "pos" in label:
                pos += score
    # normalize to length 3 list
    total = neg + neu + pos
    if total == 0:
        # fallback: try equally splitting if unknown
        return [1/3, 1/3, 1/3]
    return [neg/total, neu/total, pos/total]

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json()
    text = data.get("text", "")
    if not text:
        return jsonify({"error": "No text provided"}), 400

    resp1 = call_hf_model(MODEL1, text)
    resp2 = call_hf_model(MODEL2, text)

    probs1 = parse_model1_output(resp1) or [1/3, 1/3, 1/3]
    probs2 = parse_model2_output(resp2) or [1/3, 1/3, 1/3]

    # convert to tensors? keep using floats and numpy-like ops to avoid torch
    import math
    combined = [(p1 + p2) / 2.0 for p1, p2 in zip(probs1, probs2)]

    # amplify middle confidence (same idea as your original power)
    powered = [math.pow(x, 1.2) for x in combined]
    s = sum(powered) or 1.0
    normalized = [x / s for x in powered]

    labels = ["negative", "neutral", "positive"]
    idx = int(max(range(3), key=lambda i: normalized[i]))
    sentiment = labels[idx]
    confidence = round(normalized[idx], 3)

    return jsonify({"sentiment": sentiment, "confidence": confidence})
