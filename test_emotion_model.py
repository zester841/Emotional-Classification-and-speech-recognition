import torch
import torchaudio
import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Processor

MODEL_PATH = "saved_model/emotion_wav2vec2"
AUDIO_FOLDER = "temp_dataset"  # Path to folder with .wav files, naming should follow RAVDESS dataset format
CSV_OUTPUT = "test.csv"
MAX_LEN_SECONDS = 4
SAMPLE_RATE = 16000
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

def extract_emotion(filename):
    try:
        parts = filename.split('.')[0].split('-')
        return EMOTION_MAP.get(parts[2], "unknown")
    except:
        return "unknown"

csv_data = []
for f in os.listdir(AUDIO_FOLDER):
    if f.endswith(".wav"):
        emotion = extract_emotion(f)
        csv_data.append({"filepath": os.path.join(AUDIO_FOLDER, f), "emotion": emotion})

df = pd.DataFrame(csv_data)
df = df[df["emotion"] != "unknown"].reset_index(drop=True)
df.to_csv(CSV_OUTPUT, index=False)

processor = Wav2Vec2Processor.from_pretrained(MODEL_PATH)
model = Wav2Vec2ForSequenceClassification.from_pretrained(MODEL_PATH).to(DEVICE)
model.eval()

emotion_classes = [
    "angry", "calm", "disgust", "fearful", "happy", "neutral", "sad", "surprised"
]

le = LabelEncoder()
le.classes_ = np.array(emotion_classes)

df = df[df["emotion"].isin(emotion_classes)].reset_index(drop=True)

df["label"] = le.transform(df["emotion"])


def preprocess(filepath):
    waveform, sr = torchaudio.load(filepath)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)
    else:
        waveform = waveform.squeeze()

    if sr != SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=SAMPLE_RATE)
        waveform = resampler(waveform)

    max_len = MAX_LEN_SECONDS * SAMPLE_RATE
    if len(waveform) > max_len:
        waveform = waveform[:max_len]
    else:
        waveform = torch.nn.functional.pad(waveform, (0, max_len - len(waveform)))

    return waveform

y_true = []
y_pred = []

print("Running inference...")
for i, row in df.iterrows():
    try:
        waveform = preprocess(row["filepath"])
        inputs = processor(waveform, sampling_rate=SAMPLE_RATE, return_tensors="pt")
        with torch.no_grad():
            logits = model(inputs.input_values.to(DEVICE)).logits
        pred = torch.argmax(logits, dim=-1).item()
        y_pred.append(pred)
        y_true.append(row["label"])
    except Exception as e:
        print(f"Error at index {i}: {e}")

all_labels = list(range(len(le.classes_)))

print("\nClassification Report:")
print(classification_report(
    y_true,
    y_pred,
    labels=all_labels,
    target_names=le.classes_,
    zero_division=0  
))

cm = confusion_matrix(y_true, y_pred, labels=all_labels)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', xticklabels=le.classes_, yticklabels=le.classes_, cmap="Blues")
plt.title("Confusion Matrix - Wav2Vec2")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
plt.show()
