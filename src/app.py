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

from predict import predict_from_text, predict_from_symptoms, DISEASE_LIST, SYMPTOM_COLS_LIST
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


def normalize_symptoms_with_openai(symptom_text: str) -> list[str]:
    """
    Used ONLY when our CSV/ML matcher found nothing in the raw text.
    Asks OpenAI to map the user's free text (synonyms, slang, another
    language, etc.) onto our EXACT known symptom vocabulary
    (SYMPTOM_COLS_LIST, from Training.csv), so we can re-try the real
    CSV-based model instead of guessing a disease directly.
    """
    if not OPENAI_API_KEY:
        return []

    vocab_str = ", ".join(SYMPTOM_COLS_LIST)

    prompt = f"""
A user describes symptoms, possibly using synonyms, slang, or another
language:

\"\"\"{symptom_text}\"\"\"

Here is our EXACT list of known symptom names:

{vocab_str}

Map the user's description to the matching symptom names from that list
ONLY. Do not invent new names and do not include anything not in the list.
If nothing in the list matches, return NONE.

Return ONLY a comma-separated list of matching symptom names exactly as
spelled in the list above, or the single word NONE.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You only return exact items from the provided list, comma-separated, or NONE.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=100,
        )

        text = resp.choices[0].message.content.strip()
        if not text or text.strip().upper() == "NONE":
            return []

        candidates = [c.strip() for c in text.split(",") if c.strip()]
        # keep only symptoms that really exist in our vocabulary
        return [c for c in candidates if c in SYMPTOM_COLS_LIST]

    except Exception as e:
        print("OpenAI symptom normalization error:", e)
        return []


def get_ai_insight_for_prediction(disease: str, symptoms: list[str]) -> dict:
    """
    Even when OUR CSV/ML model already found a disease, still ask OpenAI
    for a short plain-language insight on that specific prediction, plus
    a couple of extra practical tips. This is additive context on top of
    the model's own result, never a replacement diagnosis.
    """
    if not OPENAI_API_KEY:
        return {"ai_explanation": "", "ai_suggestions": []}

    symptoms_str = ", ".join(s.replace("_", " ") for s in symptoms) or "the symptoms described"

    prompt = f"""
Our internal model predicted "{disease}" as a possible condition based on
these matched symptoms: {symptoms_str}.

In plain language:
1. Briefly explain (2-3 sentences) what this condition generally is and
   why these symptoms can relate to it. Make clear this came from a
   limited internal model, not a doctor's diagnosis.
2. Give 2-3 short, practical tips (home care, when to see a doctor, warning
   signs to watch for).

Return plain text: first the explanation, then the tips as short lines
starting with '-'.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a cautious medical information assistant giving extra "
                        "context on a prediction that already came from another model. "
                        "You never present this as a confirmed diagnosis, and you always "
                        "recommend consulting a real doctor for anything serious."
                    ),
                },
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_tokens=250,
        )

        text = resp.choices[0].message.content.strip()
        lines = [l.strip() for l in text.split("\n") if l.strip()]

        suggestions = [l.lstrip("-•* ").strip() for l in lines if l.lstrip().startswith(("-", "•", "*"))]
        explanation_lines = [l for l in lines if not l.lstrip().startswith(("-", "•", "*"))]
        explanation = " ".join(explanation_lines).strip()

        return {"ai_explanation": explanation, "ai_suggestions": suggestions}

    except Exception as e:
        print("OpenAI insight error:", e)
        return {"ai_explanation": "", "ai_suggestions": []}


def _csv_result_to_response(result: dict, normalized: bool = False) -> dict:
    """Shape a predict_from_text/predict_from_symptoms result into the API's response format."""
    insight = get_ai_insight_for_prediction(
        result["predicted_disease"], result.get("input_symptoms", [])
    )

    return {
        "ok": True,
        "predicted_disease": result["predicted_disease"],
        "ai_diseases": [d["name"] for d in result.get("possible_diseases", [])],
        "ai_explanation": insight["ai_explanation"],
        "description": result.get("description", ""),
        "precautions": result.get("precautions", []),
        "suggestions": result.get("precautions", []) or insight["ai_suggestions"],
        "ai_suggestions": insight["ai_suggestions"],
        "source": "csv_model_normalized" if normalized else "csv_model",
        "disclaimer": result.get("disclaimer", (
            "This is not a medical diagnosis. Always consult a qualified doctor "
            "for any health concerns."
        )),
    }


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    symptom_text = (data.get("symptoms") or "").strip()
    days = int(data.get("days") or 1)

    if not symptom_text:
        return jsonify({"ok": False, "message": "No symptoms provided."}), 400

    # 1) Try our own CSV/ML model first, straight from the raw text.
    csv_result = predict_from_text(symptom_text, days)
    if csv_result.get("ok"):
        return jsonify(_csv_result_to_response(csv_result)), 200

    # 2) Not matched directly — maybe the wording just doesn't match our
    #    CSV vocabulary (synonym, slang, another language). Ask OpenAI to
    #    translate it into our known symptom names, then re-try the CSV model.
    normalized_symptoms = normalize_symptoms_with_openai(symptom_text)
    if normalized_symptoms:
        csv_result = predict_from_symptoms(normalized_symptoms, days)
        if csv_result.get("ok"):
            return jsonify(_csv_result_to_response(csv_result, normalized=True)), 200

    # 3) Still nothing in our dataset — this is likely a disease/symptom
    #    outside our CSV files entirely. Fall back fully to OpenAI, which
    #    is explicitly told these are only possible conditions, not a diagnosis.
    ai = suggest_diseases_with_openai(symptom_text)

    return jsonify({
        "ok": True,
        "predicted_disease": "Not clear from internal dataset",
        "ai_diseases": ai["ai_diseases"],
        "ai_explanation": ai["ai_explanation"],
        "description": "",
        "precautions": [],
        "suggestions": ai["ai_suggestions"],
        "source": "openai_fallback",
        "disclaimer": (
            "This is not a medical diagnosis. These are only possible conditions "
            "and may be incomplete. Seek urgent care for severe chest pain, trouble "
            "breathing, fainting, weakness on one side, or worsening symptoms."
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
