
- **Multi-Modal Input**: Accept user symptoms via text entry or audio recording (using Whisper ASR).
- **Symptom Extraction**: Use a DistilBERT-based NER pipeline with a keyword fallback to identify and normalize symptom entities.
- **Condition Mapping**: Transform the wide-format `symptoms2diseases.csv` into a symptom→disease lookup and rank the top candidate conditions.
- **Urgency Classification**: Apply rule-based logic to classify cases as High, Medium, or Low urgency based on red-flag symptoms and overall symptom presence.
- **Recommendations & Self-Care**: Map urgency levels to next steps (A&E, GP booking, self-care) and surface precaution tips from `diseases2precautions.csv`.
- **Sample Dialogues**: Display example doctor replies drawn from processed MedDialog JSON entries.
- **Text-to-Speech**: Generate audio playback for precautions and doctor replies using gTTS.
- **User Interface**: Implemented with Gradio Blocks on Hugging Face Spaces, featuring automatic transcription updates and seamless analysis without extra button clicks.
