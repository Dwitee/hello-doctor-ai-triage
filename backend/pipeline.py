# backend/pipeline.py
from transformers import pipeline
import pandas as pd
import json, os
import logging
logging.basicConfig(level=logging.DEBUG)

# Automatic Speech Recognition (ASR) using Whisper
from transformers import pipeline as hf_pipeline
asr = hf_pipeline(
    "automatic-speech-recognition",
    model="openai/whisper-tiny",
    device=-1
)


import tempfile
import numpy as np
import soundfile as sf

from gtts import gTTS

def text_to_speech(text: str) -> str:
    """
    Convert text into an MP3 file using gTTS and return the file path.
    """
    tts = gTTS(text)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    tts.save(tmp.name)
    return tmp.name

def transcribe_audio(audio_input) -> str:
    """
    Transcribe the given audio (file path or numpy array) using Whisper.
    """
    logging.debug(f"transcribe_audio called with audio_input={audio_input!r}, type={type(audio_input)}")
    # Determine if input is a file path or raw audio array
    if isinstance(audio_input, str):
        logging.debug(f"transcribe_audio: detected file path, using audio_path={audio_input}")
        audio_path = audio_input
    else:
        # audio_input is a numpy array or list of audio samples
        arr = np.array(audio_input)
        logging.debug(f"transcribe_audio: detected raw array of shape {arr.shape if hasattr(arr, 'shape') else 'unknown'}, writing temp WAV")
        # Write to a temporary WAV file for ASR
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        sf.write(tmp.name, arr, 16000)
        logging.debug(f"transcribe_audio: WAV file written to {tmp.name}")
        audio_path = tmp.name
    logging.debug(f"transcribe_audio: calling ASR on {audio_path}")
    # Run the Whisper pipeline on the WAV file
    result = asr(audio_path)
    logging.debug(f"transcribe_audio: ASR result = {result}")
    return result.get("text", "").strip()

# 1) NER on CPU
ner = pipeline(
    "token-classification",
    model="distilbert-base-uncased",
    aggregation_strategy="simple",
    device=-1
)

# 2) Load lookups from CSV
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data")
# symptom → [disease, …] via melting wide CSV
wide = pd.read_csv(os.path.join(DATA_DIR, "symptoms2diseases.csv"))
symptom_cols = [col for col in wide.columns if col.lower().startswith("symptom")]
long = wide.melt(
    id_vars=["Disease"],
    value_vars=symptom_cols,
    var_name="symptom_num",
    value_name="Symptom"
)
long = long.dropna(subset=["Symptom"])
long["Disease"] = long["Disease"].str.strip()
long["Symptom"] = long["Symptom"].str.strip().str.lower()
# Normalize symptom keys: replace spaces with underscores
long["Symptom"] = long["Symptom"].str.replace(r"\s+", "_", regex=True)
sym2dis = (
    long.groupby("Symptom")["Disease"]
    .apply(list)
    .to_dict()
)
# disease → [precaution, …] via melting wide CSV
wide_p = pd.read_csv(os.path.join(DATA_DIR, "diseases2precautions.csv"))
# Identify all precaution columns
prec_cols = [col for col in wide_p.columns if col.lower().startswith("precaution")]
# Melt into long form
long_p = wide_p.melt(
    id_vars=["Disease"],
    value_vars=prec_cols,
    var_name="prec_num",
    value_name="Precaution"
)
# Drop empty entries and clean whitespace
long_p = long_p.dropna(subset=["Precaution"])
long_p["Disease"]    = long_p["Disease"].str.strip()
long_p["Precaution"] = long_p["Precaution"].str.strip()
# Build the disease→precautions mapping
dis2precs = (
    long_p.groupby("Disease")["Precaution"]
    .apply(list)
    .to_dict()
)

# 3) Load dialogs and extract doctor responses
with open(os.path.join(DATA_DIR, "dialogs.json")) as fh:
    raw_dialogs = json.load(fh)
dialogs = []
for entry in raw_dialogs:
    desc = entry.get("description", "").strip()
    # pick the first doctor utterance
    doc_utt = next((u for u in entry.get("utterances", []) if u.lower().startswith("doctor:")), "")
    if doc_utt:
        response = doc_utt.split(":", 1)[1].strip()
        dialogs.append({"input_text": desc, "response": response})

# 4) Symptom extraction
def extract_symptoms(text):
    logging.debug(f"extract_symptoms called with text: {text!r}")
    ents = ner(text)
    logging.debug(f"extract_symptoms: raw entities = {ents}")
    symptoms = []
    # Collect NER-based symptoms
    for e in ents:
        if e.get("entity_group") == "SYMPTOM":
            w = e["word"].lower().strip().replace(" ", "_")
            symptoms.append(w)
    # Fallback: keyword match against our sym2dis keys
    if not symptoms:
        logging.debug("extract_symptoms: NER found no symptoms, falling back to keyword lookup")
        text_lower = text.lower()
        for sym in sym2dis.keys():
            if sym in text_lower:
                symptoms.append(sym)
        logging.debug(f"extract_symptoms: fallback symptoms = {symptoms}")
    else:
        logging.debug(f"extract_symptoms: normalized symptoms = {symptoms}")
    return symptoms

# 5) Map symptoms → top diseases
def map_diseases(symptoms, top_k=3):
    logging.debug(f"map_diseases called with symptoms: {symptoms}")
    scores = {}
    for s in symptoms:
        logging.debug(f"map_diseases: looking up symptom key '{s}'")
        logging.debug(f"map_diseases: diseases found = {sym2dis.get(s, [])}")
    for s in symptoms:
        for d in sym2dis.get(s, []):
            scores[d] = scores.get(d, 0) + 1
    logging.debug(f"map_diseases: symptom scores = {scores}")
    ranked = sorted(scores, key=scores.get, reverse=True)
    logging.debug(f"map_diseases: ranked diseases = {ranked}")
    return ranked[:top_k] if ranked else ["Unknown"]

# 6) Urgency classification
URGENT = {
    "chest_pain",
    "difficulty_breathing",
    "stroke",
    "severe_bleeding",
    "unconsciousness",
    "seizure",
    "sudden_weakness",
    "slurred_speech",
    "severe_allergic_reaction"
}
def classify_urgency(symptoms):
    logging.debug(f"classify_urgency called with symptoms: {symptoms}")
    if URGENT & set(symptoms):
        result = "High"
    elif symptoms:
        result = "Medium"
    else:
        result = "Low"
    logging.debug(f"classify_urgency: result = {result}")
    return result
    
# 7) Recommendation assembly
ACTIONS = {
    "High": "🚨 Go to A&E immediately",
    "Medium": "📅 Book a GP within 24 hrs",
    "Low": "🛌 Self-care & pharmacist advice"
}
def recommend(disease, urgency):
    precs  = dis2precs.get(disease, [])
    # Safely handle dialogs with either 'input_text' or 'description'
    sample = next(
        (
            d.get("response", "")
            for d in dialogs
            if disease.lower() in (d.get("input_text") or d.get("description", "")).lower()
        ),
        ""
    )
    # Fallback to the first available doctor reply if no disease-specific example
    if not sample and dialogs:
        sample = dialogs[0].get("response", "")
    return {
        "disease":      disease,
        "urgency":      urgency,
        "action":       ACTIONS.get(urgency, ACTIONS["Low"]),
        "precautions":  precs,
        "sample_reply": sample or "No sample available."
    }