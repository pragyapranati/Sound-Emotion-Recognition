import os
import librosa
import numpy as np
import pandas as pd
import pickle

# Paths (FIXED)
DATASET_PATH = r"C:\Users\ASUS\Desktop\mcode\processed_data"
FEATURES_OUTPUT = r"C:\Users\ASUS\Desktop\mcode\feature_extraction"

# Ensure output directory exists
if not os.path.exists(FEATURES_OUTPUT):
    os.makedirs(FEATURES_OUTPUT)

print("Feature extraction directory set up correctly!")

# Check if metadata file exists
metadata_file = os.path.join(DATASET_PATH, "metadata.csv")
if not os.path.exists(metadata_file):
    print(f"Error: {metadata_file} not found!")
    exit()

# Function to extract features
def extract_features(file_path):
    y, sr = librosa.load(file_path, sr=22050)
    features = {
        "mfcc": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1),
        "mel_spectrogram": librosa.feature.melspectrogram(y=y, sr=sr).mean(axis=1),
        "zcr": librosa.feature.zero_crossing_rate(y).mean(),
        "rms": librosa.feature.rms(y=y).mean(),
        "chroma": librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    }
    return np.hstack(list(features.values()))

# Load metadata
metadata = pd.read_csv(metadata_file)

# Extract and save features
feature_data = []
for _, row in metadata.iterrows():
    file_path = row["Filepath"]
    emotion = row["Emotion"]

    if os.path.exists(file_path):
        try:
            print(f"Processing file: {file_path}")  # Debugging output
            features = extract_features(file_path)
            feature_data.append([features, emotion])
        except Exception as e:
            print(f"Error processing {file_path}: {e}")

# Save extracted features
feature_file = os.path.join(FEATURES_OUTPUT, "features.pkl")
with open(feature_file, "wb") as f:
    pickle.dump(feature_data, f)

print(f"Feature extraction complete! Features saved in {feature_file}")
