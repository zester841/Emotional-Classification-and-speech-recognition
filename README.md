# 🎧 Emotion Classification from Audio using Wav2Vec2 and Deep Learning

## 📝 Project Description

This project focuses on **classifying emotions from audio** using the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset. We implement and compare a variety of classical and deep learning techniques, and finally fine-tune the **Wav2Vec2** model to achieve high performance.
The project is live on Netlify(https://emotion-classification-ravdess.streamlit.app/).

---

## 📂 Dataset

### RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)

- Audio-only `.wav` files named in the format:  
  `03-01-06-01-02-01-12.wav`  
  Each filename contains metadata including **modality, vocal channel, emotion, intensity, statement, repetition**, and **actor ID**.

- Emotion Code Mapping:
01 = neutral
02 = calm
03 = happy
04 = sad
05 = angry
06 = fearful
07 = disgust
08 = surprised

  
---

## 🧪 Preprocessing

- Convert stereo to mono
- Resample to 16kHz
- Pad/truncate to fixed length (4 seconds)
- Extract features using `librosa`:
- **MFCCs**, **Chroma**, **ZCR**, **Spectral Contrast**
- Global statistics like **loudness**, **RMS**, **SNR**, etc.

---

## 🧠 Model Pipeline

We experimented with the following models:

| Model               | Accuracy |
|---------------------|----------|
| Random Forest       | 74%      |
| CNN                 | 80%      |
| CNN + LSTM          | 82%      |
| CNN + GRU           | 83%      |
| **Fine-tuned Wav2Vec2** | **89%** |

### ✅ Final Model: Wav2Vec2

- Pretrained model: `facebook/wav2vec2-base`
- Fine-tuned for multi-class emotion classification
- Training:
- Epochs: 12
- Batch size: 1
- Learning rate: 1e-5
- Evaluation metric: weighted F1 score
- Load best model based on F1

---

## 📈 Evaluation Metrics

Final evaluation on RAVDESS validation set:

- **Accuracy**: 89%
- **Weighted F1-Score**: 88.6%
- **Per-class metrics**: Provided in classification report
- **Confusion Matrix**: Created with classification report

---

## 🔄 Inference

To test the model on your own `.wav` files (following RAVDESS filename structure), run:

```bash
python test_emotion_model.py
```

## Folder Structure

```
├── data/
│   ├── speech_actors/
│   ├── song_actors/
├── saved_model/
│   └── emotion_wav2vec2/
├── test.csv
├── test_emotion_model.py
├── app.py
└── README.md
```

## Future Work

 - Ensemble multiple deep learning models

 - Add attention mechanism to BiLSTM layers

 - Extend to multilingual or multi-modal emotion datasets
