import streamlit as st
import joblib
import numpy as np
import re
from gtts import gTTS
import tempfile
import os
from rapidfuzz import process, fuzz

# --------------------------------------------------
# LOAD MODEL & ARTIFACTS
# --------------------------------------------------

model = joblib.load("medibot_model.pkl")
label_encoder = joblib.load("medibot_label_encoder.pkl")
symptom_list = joblib.load("medibot_symptom_list.pkl")
symptom_set = set(symptom_list)

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------

st.sidebar.title("⚙️ Settings")
language = st.sidebar.radio("Voice Language", ["English", "Hindi"])
TTS_LANG = "hi" if language == "Hindi" else "en"

# --------------------------------------------------
# UTILITIES
# --------------------------------------------------

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def fuzzy_match(word, choices, threshold=85):
    match = process.extractOne(word, choices, scorer=fuzz.token_sort_ratio)
    return match[0] if match and match[1] >= threshold else None

def speak_text(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    st.audio(tmp.name, format="audio/mp3")
    try:
        os.remove(tmp.name)
    except:
        pass
# ==================================================
# TEXT TO SPEECH (LANG TOGGLE)
# ==================================================

def speak_text(text, lang="en"):
    tts = gTTS(text=text, lang=lang)
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".mp3")
    tts.save(tmp.name)
    st.audio(tmp.name, format="audio/mp3")
    try:
        os.remove(tmp.name)
    except:
        pass

# --------------------------------------------------
# NLP: SYMPTOM EXTRACTION + METADATA
# --------------------------------------------------

symptom_synonyms = {
    # --------------------
    # Cold / Flu / Fever
    # --------------------
    "cold": ["runny_nose", "cough", "congestion", "sneezing"],
    "common cold": ["runny_nose", "cough", "congestion", "sneezing"],
    "flu": ["high_fever", "body_pain", "fatigue", "cough", "chills"],
    "viral infection": ["fever", "body_pain", "fatigue"],
    "viral fever": ["fever", "body_pain", "fatigue"],
    "seasonal flu": ["fever", "cough", "body_pain"],

    "fever": ["fever"],
    "high fever": ["high_fever"],
    "very high fever": ["high_fever"],
    "mild fever": ["mild_fever"],
    "low fever": ["mild_fever"],
    "temperature": ["fever"],
    "raised temperature": ["fever"],
    "feeling hot": ["fever"],

    "shivering": ["chills"],
    "chills": ["chills"],
    "cold shivers": ["chills"],
    "feeling cold": ["chills"],

    # --------------------
    # Head / ENT
    # --------------------
    "headache": ["headache"],
    "head pain": ["headache"],
    "pain in head": ["headache"],
    "migraine": ["headache", "light_sensitivity"],
    "one side headache": ["headache"],
    "pressure in head": ["headache"],

    "runny nose": ["runny_nose"],
    "running nose": ["runny_nose"],
    "nose dripping": ["runny_nose"],
    "watery nose": ["runny_nose"],

    "blocked nose": ["congestion"],
    "stuffy nose": ["congestion"],
    "nose blocked": ["congestion"],
    "nasal congestion": ["congestion"],

    "sore throat": ["sore_throat"],
    "throat pain": ["sore_throat"],
    "pain while swallowing": ["sore_throat"],
    "scratchy throat": ["sore_throat"],

    # --------------------
    # Cough / Chest
    # --------------------
    "cough": ["cough"],
    "persistent cough": ["cough"],
    "continuous cough": ["cough"],

    "dry cough": ["dry_cough"],
    "non productive cough": ["dry_cough"],

    "wet cough": ["cough"],
    "productive cough": ["cough"],

    "chest pain": ["chest_pain"],
    "pain in chest": ["chest_pain"],
    "chest discomfort": ["chest_discomfort"],
    "heaviness in chest": ["chest_discomfort"],
    "tightness in chest": ["chest_discomfort"],

    # --------------------
    # Stomach / Digestion
    # --------------------
    "stomach pain": ["stomach_pain"],
    "abdominal pain": ["stomach_pain"],
    "pain in stomach": ["stomach_pain"],
    "belly pain": ["stomach_pain"],

    "stomach cramps": ["stomach_cramps"],
    "cramps": ["stomach_cramps"],
    "abdominal cramps": ["stomach_cramps"],

    "burning in chest": ["heartburn"],
    "burning sensation": ["heartburn"],
    "acidity": ["heartburn", "stomach_burning"],
    "acid reflux": ["heartburn"],
    "gastric problem": ["heartburn"],

    "gas": ["bloating"],
    "gas problem": ["bloating"],
    "bloating": ["bloating"],
    "stomach bloating": ["bloating"],
    "feeling gassy": ["bloating"],

    "vomiting": ["vomiting"],
    "throwing up": ["vomiting"],
    "puking": ["vomiting"],

    "nausea": ["nausea"],
    "feeling nauseous": ["nausea"],
    "feeling like vomiting": ["nausea"],

    "loose motion": ["diarrhea"],
    "loose motions": ["diarrhea"],
    "diarrhea": ["diarrhea"],
    "diarrhoea": ["diarrhea"],
    "frequent stools": ["diarrhea"],
    "watery stool": ["diarrhea"],

    # --------------------
    # Body / Energy
    # --------------------
    "tired": ["fatigue"],
    "tiredness": ["fatigue"],
    "fatigue": ["fatigue"],
    "exhausted": ["fatigue"],
    "no energy": ["fatigue"],

    "body pain": ["body_pain"],
    "body ache": ["body_pain"],
    "muscle pain": ["body_pain"],
    "muscle ache": ["body_pain"],

    "weakness": ["low_energy"],
    "low energy": ["low_energy"],
    "feeling weak": ["low_energy"],
    "loss of strength": ["low_energy"],

    "joint pain": ["joint_pain"],
    "knee pain": ["joint_pain"],
    "elbow pain": ["joint_pain"],
    "joint ache": ["joint_pain"],

    # --------------------
    # Skin / Allergy
    # --------------------
    "rash": ["rash", "skin_rash"],
    "skin rash": ["skin_rash"],
    "red rash": ["skin_rash"],
    "skin redness": ["skin_rash"],

    "itching": ["itching"],
    "itchy skin": ["itching"],
    "skin itching": ["itching"],
    "pruritus": ["itching"],

    "allergy": ["sneezing", "runny_nose", "itchy_eyes", "skin_rash"],
    "dust allergy": ["sneezing", "runny_nose"],
    "pollen allergy": ["sneezing", "itchy_eyes"],
    "skin allergy": ["skin_rash", "itching"],

    # --------------------
    # COVID-related
    # --------------------
    "loss of smell": ["loss_of_smell"],
    "cannot smell": ["loss_of_smell"],
    "loss of taste": ["loss_of_taste"],
    "cannot taste": ["loss_of_taste"],
}



symptom_severity = {
    "mild_fever": 1, "fever": 2, "high_fever": 3,
    "headache": 1.5, "body_pain": 2, "fatigue": 1.5, "low_energy": 1.5,
    "runny_nose": 0.5, "sneezing": 0.5, "congestion": 1,
    "cough": 1.5, "dry_cough": 1.5, "sore_throat": 1,
    "chills": 2, "vomiting": 2, "nausea": 1.5,
    "diarrhea": 2, "stomach_pain": 2, "stomach_cramps": 2,
    "bloating": 1, "heartburn": 1.5, "stomach_burning": 1.5,
    "skin_rash": 1, "rash": 1, "itching": 0.5,
    "joint_pain": 2, "chest_discomfort": 3, "chest_pain": 4,
    "loss_of_smell": 2, "loss_of_taste": 2
}



# --------------------------------------------------
# NLP EXTRACTION (MISSPELLING-AWARE)
# --------------------------------------------------

def extract_symptoms_and_metadata(text):
    text = clean_text(text)
    words = text.split()
    found = set()

    for phrase, mapped in symptom_synonyms.items():
        if phrase in text:
            found.update(mapped)
        else:
            for w in words:
                if fuzzy_match(w, [phrase]):
                    found.update(mapped)

    for w in words:
        m = fuzzy_match(w, symptom_list, 88)
        if m:
            found.add(m)

    severity_mod = 1.0
    if "mild" in text:
        severity_mod = 0.7
    if "severe" in text or "unbearable" in text:
        severity_mod = 1.5

    days = None
    m = re.search(r"(\d+)\s*(day|days|week|weeks)", text)
    if m:
        days = int(m.group(1)) * (7 if "week" in m.group(2) else 1)

    return list(found), severity_mod, days

# --------------------------------------------------
# PREDICTION + TRIAGE
# --------------------------------------------------

def predict_disease(symptoms):
    x = np.zeros(len(symptom_list))
    idx = {s: i for i, s in enumerate(symptom_list)}
    for s in symptoms:
        if s in idx:
            x[idx[s]] = 1
    pred = model.predict([x])[0]
    return label_encoder.inverse_transform([pred])[0]

def severity_score(symptoms, mod, days):
    score = sum(symptom_severity.get(s, 1) for s in symptoms)
    if days:
        score *= 1.2 if days >= 3 else 1
    return score * mod

def triage(score):
    if score >= 12:
        return "🔴 Urgent Care", "Please consult a doctor within 24 hours."
    if score >= 6:
        return "🟠 Doctor Visit Advised", "A medical consultation is recommended."
    return "🟢 Home Care", "Rest, hydrate, and monitor symptoms."

# --------------------------------------------------
# MEDIBOT CORE
# --------------------------------------------------

def medibot_reply(text, context):
    new_symptoms, mod, days = extract_symptoms_and_metadata(text)

    symptoms = set(context["symptoms"])
    symptoms.update(new_symptoms)

    if not symptoms:
        return (
            "I couldn’t clearly detect symptoms. Try: *I have fever for 3 days*.",
            context,
            "कृपया अपने लक्षण स्पष्ट रूप से बताएं।"
        )

    disease = predict_disease(list(symptoms))
    score = severity_score(list(symptoms), mod, days)
    tri_label, tri_msg = triage(score)

    context.update({
        "symptoms": symptoms,
        "days": days,
        "last_disease": disease
    })

    reply = f"""
### 🧠 Conversation Memory Active

**Symptoms:** {", ".join(symptoms)}  
**Duration:** {days if days else "Not specified"}  
**Severity Score:** {score:.1f}

### 📌 Likely Condition
**{disease}**

### 🩺 Triage
**{tri_label}**  
{tri_msg}

⚠️ This is not a medical diagnosis.
"""

    speech_en = f"You may be suffering from {disease}. {tri_msg}"
    speech_hi = f"आपको संभवतः {disease} हो सकता है। {tri_msg}"

    return reply, context, speech_hi if TTS_LANG == "hi" else speech_en

# --------------------------------------------------
# STREAMLIT UI
# --------------------------------------------------

st.set_page_config("MediBot", "🤖")
st.title("🤖 MediBot – Intelligent Health Assistant")

if st.button("🔄 Reset Conversation"):
    st.session_state.clear()
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []
if "context" not in st.session_state:
    st.session_state.context = {"symptoms": set(), "days": None, "last_disease": None}

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Describe your symptoms...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})

    reply, ctx, speech = medibot_reply(user_input, st.session_state.context)
    st.session_state.context = ctx

    st.session_state.messages.append({"role": "assistant", "content": reply})

    with st.chat_message("assistant"):
        st.markdown(reply)
        speak_text(speech, TTS_LANG)
