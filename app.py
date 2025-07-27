# app.py
import logging
import gradio as gr
from backend.pipeline import extract_symptoms, map_diseases, classify_urgency, recommend, transcribe_audio, text_to_speech


logging.basicConfig(level=logging.DEBUG)

# Wrapper for transcribing and logging
def transcribe_and_log(audio):
    text = transcribe_audio(audio)
    logging.debug(f"transcribe_and_log: transcribed text = {text!r}")
    return text

def triage(text):
    logging.debug(f"triage: using text: {text!r}")
    syms = extract_symptoms(text)
    top  = map_diseases(syms)[0] if syms else "Unknown"
    urg  = classify_urgency(syms)
    if urg == "High":
        urg_html = '<span style="color:red">High</span>'
    elif urg == "Medium":
        urg_html = '<span style="color:orange">Medium</span>'
    else:
        urg_html = '<span style="color:green">Low</span>'
    rec = recommend(top, urg)
    precautions_text = "\n".join(f"- {p}" for p in rec["precautions"]) or "–"
    sample_text      = rec["sample_reply"] or "No sample available."
    prec_audio_path  = text_to_speech(precautions_text)
    sample_audio_path= text_to_speech(sample_text)
    return (
        text,
        ", ".join(syms) or "–",
        top,
        urg_html,
        rec["action"],
        precautions_text,
        sample_text,
        prec_audio_path,
        sample_audio_path
    )

with gr.Blocks() as demo:
    gr.HTML("<h1 style='text-align: center;'>HelloDoctor: AI-Powered Symptom Triage</h1>")
    gr.Markdown("Speak or type your symptoms below:")
    audio_input = gr.Microphone(type="filepath", label="Record Symptoms (optional)")
    text_input  = gr.Textbox(lines=4, placeholder="Describe your symptoms…", label="Symptoms")
    analyze_btn = gr.Button("Analyze")
    # When audio recording stops, transcribe and update the text box
    audio_input.change(fn=transcribe_and_log, inputs=audio_input, outputs=text_input)
    # Outputs
    out_text = text_input
    out_syms = gr.Textbox(label="Extracted Symptoms")
    out_top  = gr.Textbox(label="Top Condition")
    out_urg  = gr.HTML(label="Urgency")
    out_rec  = gr.Textbox(label="Recommendation")
    out_prec = gr.Textbox(label="Precautions")
    out_samp = gr.Textbox(label="Sample Doctor Reply")
    prec_audio   = gr.Audio(label="Play Precautions")
    sample_audio = gr.Audio(label="Play Doctor Reply")

    analyze_btn.click(
        fn=triage,
        inputs=[text_input],
        outputs=[out_text, out_syms, out_top, out_urg, out_rec, out_prec, out_samp, prec_audio, sample_audio]
    )

if __name__ == "__main__":
    demo.launch()