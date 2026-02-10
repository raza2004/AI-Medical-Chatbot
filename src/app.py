import os
import sys
import requests
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

# ✅ ADD THESE LINES - Get the correct base directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # Parent of src/
sys.path.insert(0, BASE_DIR)  # Add to Python path

from predict import predict_from_text, DISEASE_LIST
from openai import OpenAI

load_dotenv()

# ✅ UPDATE DATA_FILE path to use BASE_DIR
DATA_FILE = os.path.join(BASE_DIR, "data.json")

with open(DATA_FILE, "r", encoding="utf-8") as f:
    ALL_DOCTORS = json.load(f)
    AVAILABLE_SPECIALTIES = sorted({
        d["specialty"].strip().lower()
        for d in ALL_DOCTORS
        if d.get("specialty")
    })



LOCATIONIQ_API_KEY = os.getenv("LOCATIONIQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
CORS(app)

def analyze_symptoms_with_openai(symptom_text: str) -> dict:
    """
    Use OpenAI to give general guidance when our CSV-based model
    cannot confidently match a disease. This MUST NOT give a diagnosis.
    """
    if not OPENAI_API_KEY:
        return {
            "ai_explanation": "",
            "ai_suggestions": [],
        }

    prompt = f"""
A user describes these symptoms:

\"\"\"{symptom_text}\"\"\"

You are an AI medical information assistant. You must:
- NOT give a diagnosis or say what disease they 'have'
- Only talk about POSSIBLE general categories (like infection, allergy, etc.) in very tentative language
- Suggest very general home-care advice (rest, hydration, monitoring symptoms) if appropriate
- Clearly tell them to see a doctor or emergency services if the symptoms sound serious
- Keep it short (around 3–6 sentences)

After that short explanation, give 3–5 bullet-point style practical suggestions in plain text.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious medical information assistant. "
                        "You never give a diagnosis, never name a specific disease as certain, "
                        "and you always recommend consulting a real doctor, especially "
                        "if symptoms are severe or worsening."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=300,
        )

        text = resp.choices[0].message.content.strip()

        # very simple split: first paragraph vs bullet-ish lines
        lines = [l.strip() for l in text.split("\n") if l.strip()]
        explanation = lines[0] if lines else ""
        suggestions = []

        for line in lines[1:]:
            # remove leading bullets if present
            cleaned = line.lstrip("-•* ").strip()
            if cleaned:
                suggestions.append(cleaned)

        return {
            "ai_explanation": explanation,
            "ai_suggestions": suggestions,
        }
    except Exception as e:
        print("OpenAI fallback error:", e)
        return {
            "ai_explanation": "",
            "ai_suggestions": [],
        }
# ----------- Flask API Endpoints -----------

def suggest_diseases_with_openai(symptom_text: str) -> dict:
    """
    Use OpenAI to suggest 1–3 POSSIBLE diseases from DISEASE_LIST.
    This is NOT a diagnosis. It must be phrased as 'possible' only.
    """
    if not OPENAI_API_KEY:
        return {
            "ai_diseases": [],
            "ai_explanation": "",
            "ai_suggestions": [],
        }

    diseases_str = ", ".join(DISEASE_LIST)

    prompt = f"""
You are a cautious medical information assistant.

The user describes these symptoms:

\"\"\"{symptom_text}\"\"\"

You also have a list of diseases from a small academic dataset:

{diseases_str}

Your tasks:
1. Choose up to 3 diseases from that list that COULD POSSIBLY match the symptoms. If none match well, say so.
2. Be very clear that these are only possible conditions, not a diagnosis.
3. Give a short explanation (2–4 sentences).
4. Then give 3–5 short, practical precaution/advice bullet points (hydration, rest, seeing a doctor, warning signs).

Return your answer in this plain-text format:

Possible diseases: name1, name2 (or 'None clearly matches')
Explanation: ...
Suggestions:
- ...
- ...
- ...
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious medical information assistant. "
                        "You NEVER give a diagnosis, only possible conditions, "
                        "and you always recommend consulting a real doctor, "
                        "especially if symptoms are severe or worsening."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=400,
        )

        text = resp.choices[0].message.content.strip()

        # Very simple parsing
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        possible_line = next((l for l in lines if l.lower().startswith("possible diseases:")), "")
        explanation_line = next((l for l in lines if l.lower().startswith("explanation:")), "")

        # suggestions lines after "Suggestions:"
        suggestions_start = None
        for i, l in enumerate(lines):
            if l.lower().startswith("suggestions:"):
                suggestions_start = i + 1
                break

        suggestions = []
        if suggestions_start is not None:
            for l in lines[suggestions_start:]:
                if l.lower().startswith("possible diseases:") or l.lower().startswith("explanation:"):
                    break
                cleaned = l.lstrip("-•* ").strip()
                if cleaned:
                    suggestions.append(cleaned)

        # parse diseases
        ai_diseases = []
        if possible_line:
            raw = possible_line.split(":", 1)[-1]
            ai_diseases = [d.strip() for d in raw.split(",") if d.strip()]

        explanation = explanation_line.split(":", 1)[-1].strip() if explanation_line else ""

        return {
            "ai_diseases": ai_diseases,
            "ai_explanation": explanation,
            "ai_suggestions": suggestions,
        }

    except Exception as e:
        print("OpenAI disease suggestion error:", e)
        return {
            "ai_diseases": [],
            "ai_explanation": "",
            "ai_suggestions": [],
        }


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "HealthNexus backend running"})


def is_clear_from_dataset(symptom_text: str) -> bool:
    # Simple heuristic: if user typed very few words, or text is too vague → not clear
    t = (symptom_text or "").strip().lower()
    if len(t.split()) < 3:
        return False
    return True  # replace later with a real confidence score if you add one


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    symptom_text = (data.get("symptoms") or "").strip()

    # 1) If internal dataset not clear → AI-only (possible conditions)
    if not is_clear_from_dataset(symptom_text):
        ai = suggest_diseases_with_openai(symptom_text)

        return jsonify({
            "ok": True,
            "predicted_disease": "Not clear from internal dataset",
            "ai_diseases": ai["ai_diseases"],               # ✅ show these in UI
            "ai_explanation": ai["ai_explanation"],
            "description": "",
            "precautions": [],
            "suggestions": ai["ai_suggestions"],
            "disclaimer": (
                "This is not a medical diagnosis. These are only possible conditions "
                "and may be incomplete. Seek urgent care for severe chest pain, trouble "
                "breathing, fainting, weakness on one side, or worsening symptoms."
            ),
        }), 200

    # 2) Otherwise: use your existing logic (AI picks from DISEASE_LIST)
    ai = suggest_diseases_with_openai(symptom_text)
    primary_disease = ai["ai_diseases"][0] if ai["ai_diseases"] else None

    description, precautions = "", []
    if primary_disease:
        from predict import desc_dict, precaution_dict
        description = desc_dict.get(primary_disease, "")
        precautions = precaution_dict.get(primary_disease, [])

    suggestions = precautions or ai["ai_suggestions"]

    return jsonify({
        "ok": True,
        "predicted_disease": primary_disease or "Not clear from internal dataset",
        "ai_diseases": ai["ai_diseases"],
        "ai_explanation": ai["ai_explanation"],
        "description": description,
        "precautions": precautions,
        "suggestions": suggestions,
        "disclaimer": (
            "This is not a medical diagnosis. These are only possible conditions "
            "from a limited dataset. Always consult a qualified doctor."
        ),
    }), 200


# --- helper: ask OpenAI for specialty / guidance text (optional) ---
def build_doctor_advice(disease: str) -> str:
    """
    Uses OpenAI only for text guidance, not for searching doctors.
    """
    if not OPENAI_API_KEY:
        return ""

    prompt = f"""
Given the disease name: "{disease}", briefly answer:

1. What type of doctor or specialist people usually consult for this?
2. Very short general advice (non-diagnostic, always recommending seeing a real doctor).

Return 2–3 sentences, plain text, no markdown, no bullet points.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious medical information assistant. "
                        "You never give a diagnosis and always suggest consulting a real doctor."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=150,
        )

        text = resp.choices[0].message.content.strip()
        return text
    except Exception as e:
        print("OpenAI error:", e)
        return ""

# 
def normalize(text):
    return text.lower().strip() if text else ""

def get_specialties_for_disease(disease: str) -> list[str]:
    if not OPENAI_API_KEY:
        return ["general physician"]

    specialties_text = "\n".join(f"- {s}" for s in AVAILABLE_SPECIALTIES)

    prompt = f"""
You are a medical triage assistant.

User symptoms or disease:
"{disease}"

Choose UP TO 3 most relevant specialties
from the list below.

Rules:
- You MUST choose only from the list
- Prefer medical relevance over exact wording
- Order them from most relevant to least relevant
- If unsure, include broader specialties

Available specialties:
{specialties_text}

Return as a comma-separated list.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "Return only specialties from the list."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=50,
        )

        raw = resp.choices[0].message.content.lower()
        return [normalize(s) for s in raw.split(",") if s.strip()]

    except Exception as e:
        print("OpenAI specialty error:", e)
        return ["general physician"]

def get_doctors_from_json(city, specialty, limit=5):
    city = normalize(city)
    specialty = normalize(specialty).replace("-", " ")

    results = []

    for d in ALL_DOCTORS:
        if normalize(d.get("city")) != city:
            continue

        doc_specialty = normalize(d.get("specialty"))
        if specialty not in doc_specialty:
            continue

        results.append(d)

        if len(results) >= limit:
            break

    return results

@app.route("/doctors", methods=["POST"])
def doctors():
    data = request.get_json() or {}
    disease = (data.get("disease") or "").strip()
    city = (data.get("city") or "").strip()

    if not disease or not city:
        return jsonify({"error": "Disease and city are required"}), 400

    specialties = get_specialties_for_disease(disease)  # ✅ ALWAYS define
    print("AI-selected specialties:", specialties)

    city_norm = normalize(city)
    doctors_list = []

    for d in ALL_DOCTORS:
        if not d.get("name") or not d.get("specialty"):
            continue
        if normalize(d.get("city")) != city_norm:
            continue

        doc_specialty = normalize(d.get("specialty"))  # ✅ define before using

        # ✅ match any suggested specialty
        if not any(s in doc_specialty for s in specialties):
            continue

        doctors_list.append({
            "name": d.get("name"),
            "specialty": d.get("specialty"),
            "qualifications": d.get("qualifications") or "Not specified",
            "experience": d.get("experience") or "Not specified",
            "fee": d.get("fee") or "Not available",
            "reviews": int(d.get("reviews_count") or 0) if str(d.get("reviews_count") or "").isdigit() else 0,
            "satisfaction": d.get("satisfaction"),
            "pmdc_verified": bool(d.get("pmdc_verified")),
            "video_consultation": bool(d.get("video_consultation")),
            "url": d.get("url"),
            "city": d.get("city"),
        })

        if len(doctors_list) >= 5:
            break

    doctor_advice = build_doctor_advice(disease)

    if not doctors_list:
        return jsonify({
            "doctors": [],
            "doctor_advice": doctor_advice,
            "message": f"No doctors found in {city} for: {', '.join(specialties)}"
        }), 200

    return jsonify({
        "doctors": doctors_list,
        "doctor_advice": doctor_advice,
        "specialties": specialties,
    }), 200


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
    # ✅ ADD THIS - Required for Vercel
app = app 
