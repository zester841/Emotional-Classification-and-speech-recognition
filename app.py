import streamlit as st
import torch
import torchaudio
import numpy as np
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

MODEL_PATH = "saved_model_cpu/emotion_wav2vec2"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

emotion_labels = ["angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"]

st.set_page_config(
    page_title="Emotion Recognition from Speech",
    page_icon="🎙️",
    layout="centered",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .main-title {
        font-size:40px;
        font-weight:bold;
        color:#FF4B4B;
        text-align:center;
        margin-bottom:10px;
    }
    .subtext {
        text-align:center;
        color:gray;
        margin-bottom:20px;
    }
    .result-box {
        border-radius: 10px;
        padding: 15px;
        background-color: #f9f9f9;
        border: 1px solid #ddd;
        text-align: center;
        font-size: 22px;
        font-weight: bold;
        color: #333333;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown('<div class="main-title">Speech Emotion Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtext">Upload a WAV file and detect the underlying emotion</div>', unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose a .wav audio file", type=["wav"])

def preprocess(wav_bytes):
    waveform, sr = torchaudio.load(wav_bytes)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze()

    if sr != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=16000)
        waveform = resampler(waveform)

    max_len = 4 * 16000
    if len(waveform) > max_len:
        waveform = waveform[:max_len]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, max_len - len(waveform)))

    return waveform

if uploaded_file is not None:
    
    st.audio(uploaded_file, format='audio/wav')

    with open("temp_audio.wav", "wb") as f:
        f.write(uploaded_file.read())

    try:
        waveform = preprocess("temp_audio.wav")
        inputs = processor(waveform, sampling_rate=16000, return_tensors="pt")

        with torch.no_grad():
            logits = model(inputs.input_values.to(DEVICE)).logits
        pred_id = torch.argmax(logits, dim=-1).item()
        predicted_emotion = emotion_labels[pred_id]

        st.markdown(f'<div class="result-box">🔎 Predicted Emotion: <span style="color:#FF4B4B;">{predicted_emotion.upper()}</span></div>', unsafe_allow_html=True)

    except Exception as e:
        st.error(f"Error processing audio: {e}")
