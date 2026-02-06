import os
import requests
import re
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from apify_client import ApifyClient

from predict import predict_from_text, DISEASE_LIST

from openai import OpenAI

load_dotenv()

LOCATIONIQ_API_KEY = os.getenv("LOCATIONIQ_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
APIFY_API_TOKEN = os.getenv("APIFY_API_TOKEN")
apify_client = ApifyClient(APIFY_API_TOKEN)

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


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json() or {}
    symptom_text = data.get("symptoms", "")
    days = data.get("days", 1)

    try:
        days = int(days)
    except (TypeError, ValueError):
        days = 1

    # For this free-text interface, IGNORE the DecisionTree disease
    # and use OpenAI to suggest possible conditions from DISEASE_LIST.
    ai = suggest_diseases_with_openai(symptom_text)

    # choose primary_disease as the first AI suggestion, if any
    primary_disease = ai["ai_diseases"][0] if ai["ai_diseases"] else None

    # lookup description/precautions from CSV-based dicts (if disease name matches)
    description = ""
    precautions = []

    if primary_disease:
        from predict import desc_dict, precaution_dict  # or expose them similarly
        description = desc_dict.get(primary_disease, "")
        precautions = precaution_dict.get(primary_disease, [])

    # if CSV has no precautions, fallback to AI suggestions
    suggestions = precautions or ai["ai_suggestions"]

    result = {
        "ok": True,
        "predicted_disease": primary_disease,
        "ai_diseases": ai["ai_diseases"],
        "ai_explanation": ai["ai_explanation"],
        "description": description,
        "precautions": precautions,
        "suggestions": suggestions,
        "disclaimer": (
            "This is not a medical diagnosis. These are only possible conditions "
            "from a limited dataset. Always consult a qualified doctor for any "
            "health concerns, especially for severe or persistent symptoms."
        ),
    }

    return jsonify(result), 200

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
def get_specialty_for_disease(disease: str) -> str:
    """
    Use OpenAI to map ANY disease name (even unfamiliar ones)
    to the closest medical specialty used on Marham.pk.
    """
    if not OPENAI_API_KEY:
        return "general-physician"

    prompt = f"""
You are a medical triage assistant.

Given a disease name or symptom cluster:

"{disease}"

Return ONLY the most appropriate specialist type from this list
(found on Marham.pk):

- acupuncture
- aesthetic-physician
- allergy-specialist
- andrologist
- anesthetist
- audiologist
- bariatric-surgeon
- cancer-surgeon
- cardiac-surgeon
- cardiologist
- chest-surgeon
- child-psychologist
- chiropractor
- clinical-nutritionist
- clinical-psychologist
- cosmetic-surgeon
- counselor
- dentist
- dermatologist
- diabetologist
- endocrinologist
- endourologist
- ent-specialist
- ent-surgeon
- eye-specialist
- eye-surgeon
- family-medicine
- gastroenterologist
- general
- general-physician
- general-practitioner
- general-surgeon
- gynecologist
- hematologist
- hepatologist
- hijama-specialist
- homeopath
- infectious-diseases
- internal-medicine
- interventional-cardiologist
- interventional-radiologist
- laproscopic-surgeon
- liver-specialist
- liver-transplant-surgeon
- lung-surgeon
- maternal-fetal-medicine-specialist
- maxillofacial-surgeon
- medical-specialist
- neonatologist
- nephrologist
- neuro-physician
- neuro-psychiatrist
- neuro-surgeon
- nutritionist
- oncologist
- optometrist
- orthodontist
- orthopedic-surgeon
- pain-specialist
- pathologist
- pediatric-cardiac-surgeon
- pediatric-cardiologist
- pediatric-endocrinologist
- pediatric-gastroenterologist
- pediatric-nephrologist
- pediatric-neuro-physician
- pediatric-neurosurgeon
- pediatric-oncologist-hematologist
- pediatric-orthopedic-surgeon
- pediatric-surgeon
- pediatrician
- physiotherapist
- plastic-surgeon
- psychiatrist
- psychologist
- pulmonologist
- radiation-oncologist
- radiologist
- regenerative-medicine-specialist
- restorative-dentist
- rheumatologist
- sexologist
- sonologist
- speech-therapist
- urological-oncologist
- urologist
- vaccine-specialist
- vascular-surgeon

Return ONLY the specialty slug EXACTLY as written above.
Do NOT explain anything. Do NOT add extra words.
"""

    try:
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You return ONLY a specialty slug."},
                {"role": "user", "content": prompt},
            ],
            temperature=0,
            max_tokens=10,
        )

        specialty = resp.choices[0].message.content.strip().lower()

        # Clean formatting just in case
        specialty = specialty.replace(" ", "-")

        return specialty

    except Exception as e:
        print("OpenAI specialty error:", e)
        return "general-physician"

@app.route("/doctors", methods=["POST"])
def doctors():
    data = request.get_json() or {}
    disease = (data.get("disease") or "").strip()
    city = (data.get("city") or "").strip()

    if not disease or not city:
        return jsonify({"error": "Disease and city are required"}), 400

    # 1️⃣ Use OpenAI to map disease → specialty
    specialty = get_specialty_for_disease(disease)
    print("AI-selected specialty:", specialty)

    # 2️⃣ Call Apify Marham Actor with the CORRECT input format
    try:
        run = apify_client.actor("shahidirfan/marham-pk-scraper").call(
            run_input={
                "city": city.lower(),
                "specialty": specialty.lower(),
                "results_wanted": 5,
                "proxyConfiguration": {
                    "useApifyProxy": True,
                    "apifyProxyGroups": ["RESIDENTIAL"]
                }
            }
        )

        dataset_id = run["defaultDatasetId"]
        items = list(apify_client.dataset(dataset_id).iterate_items())

    except Exception as e:
        print("Apify error:", e)
        return jsonify({
            "doctors": [],
            "doctor_advice": build_doctor_advice(disease),
            "message": "Failed to fetch doctor data from Apify."
        }), 200

    # 3️⃣ No doctors returned
    if not items:
        return jsonify({
            "doctors": [],
            "doctor_advice": build_doctor_advice(disease),
            "message": f"No {specialty} doctors found in {city}."
        }), 200

    # 4️⃣ Format doctor data (Apify data already clean)
    doctors_list = []
    for d in items:
        doctors_list.append({
            "name": d.get("name"),
            "specialty": d.get("specialty"),
            "qualifications": d.get("qualifications"),
            "experience": d.get("experience"),
            "fee": d.get("fee"),
            "satisfaction": d.get("satisfaction"),
            "reviews": d.get("reviews_count"),
            "pmdc_verified": d.get("pmdc_verified"),
            "video_consultation": d.get("video_consultation"),
            "url": d.get("url"),
            "city": d.get("city") or city,
        })

    # 5️⃣ AI doctor advice based on disease
    doctor_advice = build_doctor_advice(disease)

    return jsonify({
        "doctors": doctors_list,
        "doctor_advice": doctor_advice,
        "specialty": specialty,
    }), 200

    data = request.get_json() or {}
    disease = (data.get("disease") or "").strip()
    city = (data.get("city") or "").strip()

    if not disease or not city:
        return jsonify({"error": "Disease and city are required"}), 400

    # 1️⃣ Map disease → specialty using OpenAI
    specialty = get_specialty_for_disease(disease)
    print("AI-selected specialty:", specialty)

    # 2️⃣ Call Apify Marham Scraper Actor
    try:
        run = apify_client.actor("shahidirfan/marham-pk-scraper").call(
            run_input={
                "specialty": specialty,
                "city": city,
                "results_wanted": 5,
                # "collectDetails": True,
                "useJsonApi": True
            }
        )

        dataset_id = run["defaultDatasetId"]

        # Fetch the dataset items
        items = list(apify_client.dataset(dataset_id).iterate_items())

    except Exception as e:
        print("Apify error:", e)
        return jsonify({
            "doctors": [],
            "doctor_advice": build_doctor_advice(disease),
            "message": "Failed to fetch doctor data from Apify."
        }), 200

    # 3️⃣ If Apify returned nothing
    if not items:
        return jsonify({
            "doctors": [],
            "doctor_advice": build_doctor_advice(disease),
            "message": f"No {specialty} doctors found in {city}."
        }), 200

    # 4️⃣ Build the doctor response cleanly
    doctors_list = []
    for doc in items:
        doctors_list.append({
            "name": doc.get("name"),
            "specialty": doc.get("specialty"),
            "qualifications": doc.get("qualifications"),
            "experience": doc.get("experience"),
            "fee": doc.get("fee"),
            "reviews": doc.get("reviews_count"),
            "rating": doc.get("satisfaction"),
            "hospitals": doc.get("hospitals"),
            "services": doc.get("services"),
            "about": doc.get("about"),
            "city": doc.get("city") or city,
            "url": doc.get("url"),
            "video_consultation": doc.get("video_consultation"),
            "pmdc_verified": doc.get("pmdc_verified"),
        })

    # 5️⃣ Build AI doctor advice
    doctor_advice = build_doctor_advice(disease)

    return jsonify({
        "doctors": doctors_list,
        "doctor_advice": doctor_advice,
        "specialty": specialty,
    }), 200


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
