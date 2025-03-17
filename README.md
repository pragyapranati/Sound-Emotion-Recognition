# 🎙️ Real-Time Speech Emotion Recognition

## 🌟 Overview

Welcome to the **Real-Time Speech Emotion Recognition** project! This repository presents an advanced deep learning-based approach for detecting human emotions through speech signals using **Convolutional Neural Networks (CNN)** and **Bidirectional Long Short-Term Memory (BiLSTM)** models. The system utilizes **data augmentation techniques** to enhance accuracy and robustness in real-world applications.

---

## 📌 Features
- 🧠 **Deep Learning Models**: CNN, MLP, and CNN+BiLSTM.
- 🎵 **Feature Extraction**: MFCC, Mel Spectrogram, ZCR, RMS, Chroma.
- 🎤 **Dataset Utilization**: RAVDESS, TESS, and EmoDB.
- 🔊 **Data Augmentation**: Noise Addition & Spectrogram Shift.
- 🚀 **Real-Time Processing** for accurate emotion recognition.

---

## 🏗️ Project Architecture
```
📁 Real-Time-SER
├── 📂 datasets/         # Processed Speech Emotion Datasets
├── 📂 models/           # Pretrained and Fine-tuned Deep Learning Models
├── 📂 scripts/          # Data Preprocessing and Feature Extraction Scripts
├── 📂 notebooks/        # Jupyter Notebooks for Model Training & Evaluation
├── README.md           # Project Documentation
└── requirements.txt     # Dependencies & Libraries
```

---

## 🎯 Model Performance
| Model         | TESS Accuracy | EmoDB Accuracy | RAVDESS Accuracy |
|--------------|--------------|---------------|----------------|
| **MLP**      | 99.90%       | 98.76%        | 90.09%         |
| **CNN**      | 99.95%       | 98.51%        | 86.85%         |
| **CNN+BiLSTM** | **100%**    | **99.50%**   | **90.12%**    |

---

## 📖 Research Paper
For in-depth details about our approach, you can read the full research paper published in **Artificial Intelligence Review (2025)**:
🔗 **[Real-time Speech Emotion Recognition using Deep Learning and Data Augmentation](https://doi.org/10.1007/s10462-024-11065-x)**

---

## 🛠️ Installation
### 1️⃣ Clone this Repository
```sh
git clone https://github.com/yourusername/Real-Time-SER.git
cd Real-Time-SER
```

### 2️⃣ Install Dependencies
```sh
pip install -r requirements.txt
```

### 3️⃣ Run the Real-Time Recognition
```sh
python real_time_ser.py
```

🚀 **Happy Coding!**
