# --- add at top of file imports if not present ---
import os
import math
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from flask import Flask, request, jsonify, render_template

# --- session with retries (reuse existing session if you added one) ---
session = requests.Session()
retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[429,500,502,503,504])
session.mount("https://", HTTPAdapter(max_retries=retries))

HF_API_TOKEN = os.environ.get("HF_API_TOKEN", "")
HEADERS = {"Authorization": f"Bearer {HF_API_TOKEN}"} if HF_API_TOKEN else {}

MODEL1 = "cardiffnlp/twitter-roberta-base-sentiment"
MODEL2 = "nlptown/bert-base-multilingual-uncased-sentiment"
HF_API_URL = "https://api-inference.huggingface.co/models/"

def call_hf_model(model_name, text, timeout=30):
    url = HF_API_URL + model_name
    payload = {"inputs": text, "options": {"wait_for_model": True}}
    try:
        r = session.post(url, headers=HEADERS, json=payload, timeout=timeout)
        r.raise_for_status()
        j = r.json()
        # log raw HF output for debugging (appears in Vercel logs)
        print(f"[HF RAW] model={model_name} status={r.status_code} resp={j}")
        return j
    except Exception as e:
        print(f"[HF ERROR] model={model_name} err={e}")
        return None

def parse_model1_output(hf_resp):
    # cardiffnlp usually returns negative/neutral/positive or LABEL_0..LABEL_2
    if not isinstance(hf_resp, list): 
        return None
    probs = [0.0, 0.0, 0.0]
    for item in hf_resp:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
        if "neg" in label:
            probs[0] = score
        elif "neu" in label:
            probs[1] = score
        elif "pos" in label:
            probs[2] = score
        else:
            # handle LABEL_0 / LABEL_1 mapping: many HF models use LABEL_0 -> negative etc
            import re
            m = re.search(r"(\d+)", label)
            if m:
                idx = int(m.group(1))
                # try to guess ordering: if there are 3 labels, assume LABEL_0=neg,1=neu,2=pos
                if 0 <= idx < 3:
                    probs[idx] = score
    # fallback: if all zeros but list length >= 3, assign by order
    total = sum(probs)
    if total == 0.0 and len(hf_resp) >= 3:
        for i in range(3):
            probs[i] = float(hf_resp[i].get("score", 0.0))
    # final normalization guard
    s = sum(probs) or 1.0
    return [p / s for p in probs]

def parse_model2_output(hf_resp):
    # nlptown returns 1..5 stars. Map 1-2 -> neg, 3 -> neu, 4-5 -> pos
    if not isinstance(hf_resp, list):
        return None
    neg, neu, pos = 0.0, 0.0, 0.0
    for item in hf_resp:
        label = str(item.get("label", "")).lower()
        score = float(item.get("score", 0.0))
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
            # fallback: check words
            if "neg" in label:
                neg += score
            elif "neu" in label:
                neu += score
            elif "pos" in label:
                pos += score
    total = neg + neu + pos
    if total == 0.0:
        # fallback: try order
        if len(hf_resp) >= 5:
            # roughly convert 5-class to 3-class by grouping
            scores = [float(it.get("score",0.0)) for it in hf_resp]
            neg = scores[0] + scores[1]
            neu = scores[2]
            pos = sum(scores[3:5])
            total = neg + neu + pos
        else:
            total = 1.0
    return [neg/total, neu/total, pos/total]

@app.route("/analyze", methods=["POST"])
def analyze():
    data = request.get_json(force=True)
    text = data.get("text", "")
    if not text:
        return jsonify({"error":"no text provided"}), 400

    resp1 = call_hf_model(MODEL1, text)
    resp2 = call_hf_model(MODEL2, text)

    probs1 = parse_model1_output(resp1) or [1/3, 1/3, 1/3]
    probs2 = parse_model2_output(resp2) or [1/3, 1/3, 1/3]

    # Combine and normalize
    combined = [(p1 + p2) / 2.0 for p1, p2 in zip(probs1, probs2)]
    # amplify middle confidence like your original idea
    powered = [math.pow(x, 1.2) for x in combined]
    s = sum(powered) or 1.0
    normalized = [x / s for x in powered]

    labels = ["negative", "neutral", "positive"]
    idx = int(max(range(3), key=lambda i: normalized[i]))
    sentiment = labels[idx]
    confidence = round(normalized[idx], 3)

    # Return helpful debug info so you can see intermediate values in the browser
    return jsonify({
        "sentiment": sentiment,
        "confidence": confidence,
        "probs_model1": probs1,
        "probs_model2": probs2,
        "combined_raw": combined,
        "combined_after_power": normalized
    })
