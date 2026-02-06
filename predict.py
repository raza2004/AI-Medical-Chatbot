# backend/predict.py

import os
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

# ----------------- PATHS -----------------
BASE_DIR = Path(__file__).resolve().parent
FILES_DIR = BASE_DIR / "Files"

TRAINING_PATH = FILES_DIR / "Training.csv"
SEVERITY_PATH = FILES_DIR / "Symptom_severity.csv"
DESCRIPTION_PATH = FILES_DIR / "symptom_Description.csv"
PRECAUTION_PATH = FILES_DIR / "symptom_precaution.csv"

# ----------------- LOAD & TRAIN MODEL ONCE -----------------

training = pd.read_csv(TRAINING_PATH)

cols = training.columns
SYMPTOM_COLS = cols[:-1]
X = training[SYMPTOM_COLS]
y = training["prognosis"]

# list of all diseases from dataset
DISEASE_LIST = sorted(list(set(y)))

# Encode disease names to numbers
le = preprocessing.LabelEncoder()
le.fit(y)
y_encoded = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.33, random_state=42
)

clf = DecisionTreeClassifier()
clf.fit(X_train, y_train)

# For probabilities
prob_model = clf if hasattr(clf, "predict_proba") else None

# ----------------- LOAD EXTRA CSVs -----------------

severity_dict = {}
desc_dict = {}
precaution_dict = {}

def _load_severity():
    df = pd.read_csv(SEVERITY_PATH)
    # Expect columns: Symptom, weight
    for _, row in df.iterrows():
        try:
            severity_dict[row[0]] = int(row[1])
        except Exception:
            continue

def _load_descriptions():
    df = pd.read_csv(DESCRIPTION_PATH)
    # Expect columns: Disease, Description
    for _, row in df.iterrows():
        desc_dict[row[0]] = row[1]

def _load_precautions():
    df = pd.read_csv(PRECAUTION_PATH)
    # Expect columns: Disease, Precaution_1..4
    for _, row in df.iterrows():
        disease = row[0]
        precs = [p for p in row[1:] if isinstance(p, str) and p.strip()]
        precaution_dict[disease] = precs

_load_severity()
_load_descriptions()
_load_precautions()

# ----------------- HELPERS -----------------

def _calc_severity(symptoms, days):
    """
    Similar idea to calc_condition in the repo:
    larger (sum_of_severity * days) => more severe.
    """
    total = 0
    for s in symptoms:
        if s in severity_dict:
            total += severity_dict[s]

    score = (total * days) / (len(symptoms) + 1) if symptoms else 0

    if score > 13:
        msg = "You should consult a doctor."
        severe = True
    else:
        msg = "It may not be severe, but you should take precautions."
        severe = False

    return score, severe, msg

# simple manual synonym mapping (you can expand this over time)
SYMPTOM_SYNONYMS = {
    "pain in head": "headache",
    "head pain": "headache",
    "head hurts": "headache",
    "stomach ache": "stomach_pain",
    "stomachache": "stomach_pain",
    "tummy pain": "stomach_pain",
    "flu": "cold",
    "runny nose": "runny_nose" if "runny_nose" in SYMPTOM_COLS else "",
    # add more as needed based on your CSV columns
}

def _text_to_symptom_list(text: str):
    """
    Convert free-text into list of symptom column names matching Training.csv.
    """
    normalized_text = text.lower().replace("-", " ")
    normalized_text = normalized_text.replace(",", " ")

    found = set()

    # 1) direct matches using column names
    for col in SYMPTOM_COLS:
        col_words = col.replace("_", " ")
        if col_words in normalized_text:
            found.add(col)

    # 2) synonym-based matches
    for phrase, mapped_symptom in SYMPTOM_SYNONYMS.items():
        if phrase in normalized_text and mapped_symptom and mapped_symptom in SYMPTOM_COLS:
            found.add(mapped_symptom)

    print("DEBUG: input text:", normalized_text)
    print("DEBUG: matched symptoms:", found)

    return list(found)

# ----------------- PUBLIC FUNCTION -----------------

def predict_from_text(symptom_text: str, days: int = 1):
    """
    Main function to be used by Flask.
    Takes a free-text symptom description and number of days,
    returns a JSON-serializable dict with prediction info.
    """

    symptoms = _text_to_symptom_list(symptom_text)

    # Build 0/1 feature vector
    input_vec = np.zeros(len(SYMPTOM_COLS), dtype=int)
    col_list = list(SYMPTOM_COLS)

    for s in symptoms:
        if s in col_list:
            idx = col_list.index(s)
            input_vec[idx] = 1

    if not symptoms:
        # no known symptoms from our CSV
        return {
            "ok": False,
            "message": "No known symptoms found in input. Try using more medical terms.",
            "input_symptoms": [],
        }

    # If we have a prob model, use it properly
    if prob_model is not None:
        probs = prob_model.predict_proba([input_vec])[0]
        classes = le.inverse_transform(np.arange(len(probs)))
        best_idx = int(np.argmax(probs))
        best_prob = float(probs[best_idx])
        primary_disease = str(classes[best_idx])

        print("DEBUG: model probs (top few):")
        for idx in np.argsort(probs)[::-1][:5]:
            print(f"  {classes[idx]} -> {probs[idx]:.3f}")
        print("DEBUG: best disease:", primary_disease, "prob:", best_prob)

        # Confidence threshold: if too low, treat as not confident
        CONFIDENCE_THRESHOLD = 0.45
        if best_prob < CONFIDENCE_THRESHOLD:
            return {
                "ok": False,
                "message": "Symptoms did not confidently match any disease in this limited dataset.",
                "input_symptoms": symptoms,
                "model_best_guess": primary_disease,
                "model_confidence": best_prob,
            }

        # otherwise, build top-2 list
        top_diseases = []
        sorted_idx = np.argsort(probs)[::-1]
        for idx in sorted_idx[:2]:
            top_diseases.append({
                "name": str(classes[idx]),
                "probability": float(probs[idx]),
            })

    else:
        # fallback: no prob_model (unlikely since DecisionTree has it)
        y_pred = clf.predict([input_vec])[0]
        primary_disease = le.inverse_transform([y_pred])[0]
        top_diseases = [{"name": primary_disease, "probability": None}]

    # Severity & message
    severity_score, is_severe, severity_msg = _calc_severity(symptoms, days)

    # Descriptions & precautions
    description = desc_dict.get(primary_disease, "")
    precautions = precaution_dict.get(primary_disease, [])

    return {
        "ok": True,
        "input_symptoms": symptoms,
        "days": days,
        "predicted_disease": primary_disease,
        "possible_diseases": top_diseases,
        "severity_score": severity_score,
        "severity_message": severity_msg,
        "is_severe": is_severe,
        "description": description,
        "precautions": precautions,
        "disclaimer": (
            "This is not a medical diagnosis. Always consult a qualified doctor "
            "for any health concerns."
        ),
    }
# expose disease list for the app
__all__ = ["predict_from_text", "DISEASE_LIST"]
