import os
import librosa
import librosa.display
import numpy as np
import pandas as pd
import soundfile as sf
import random

# Define dataset paths
DATASET_PATHS = {
    "EmoDB": "C:\\Users\\ASUS\\Desktop\\mcode\\emoDB\wav",  
    "RAVDESS": "C:\\Users\\ASUS\\Desktop\\mcode\\ravdess_part",  
    "TESS": "C:\\Users\\ASUS\\Desktop\\mcode\\TESS Toronto emotional speech set data"
}

OUTPUT_PATH = "C:\\Users\\ASUS\\Desktop\\mcode\\processed_data"  # Folder where processed files are stored
AUGMENTED_PATH = "C:\\Users\\ASUS\\Desktop\\mcode\\augemented_data"

# Ensure output directories exist
os.makedirs(OUTPUT_PATH, exist_ok=True)
os.makedirs(AUGMENTED_PATH, exist_ok=True)

# Define EmoDB emotion mapping
EMODB_EMOTIONS = {
    "W": "anger",
    "L": "boredom",
    "E": "disgust",
    "A": "fear",
    "F": "happy",
    "N": "neutral",
    "T": "sad"
}

# Define RAVDESS emotion mapping
RAVDESS_EMOTIONS = {
    "01": "neutral", "02": "calm", "03": "happy", "04": "sad", 
    "05": "angry", "06": "fearful", "07": "disgusted", "08": "surprised"
}

# Function to extract emotion labels from filenames
def get_emotion_label(dataset, filename):
    if dataset == "EmoDB":
        emotion_code = filename[5]  # 6th character in filename
        return EMODB_EMOTIONS.get(emotion_code, "unknown")
    elif dataset == "RAVDESS":
        emotion_code = filename.split("-")[2]  # 3rd number in filename
        return RAVDESS_EMOTIONS.get(emotion_code, "unknown")
    elif dataset == "TESS":
        return filename.split("_")[2].split(".")[0].lower()  # Extract emotion name
    return "unknown"

# Function to load and preprocess audio files
def process_audio(file_path, sr=22050):
    y, sr = librosa.load(file_path, sr=sr)  # Load audio
    y = librosa.util.normalize(y)  # Normalize volume
    return y, sr

# Function to add noise
def add_noise(y, noise_level=0.005):
    return y + noise_level * np.random.randn(len(y))

# Function to shift audio
def shift_audio(y, shift_max=2, sr=22050):
    shift = np.random.randint(-shift_max * sr, shift_max * sr)
    return np.roll(y, shift)

# Process all datasets
metadata = []
for dataset, path in DATASET_PATHS.items():
    print(f"Processing {dataset} dataset...")
    for root, _, files in os.walk(path):
        for file in files:
            if file.endswith(".wav"):
                file_path = os.path.join(root, file)
                emotion = get_emotion_label(dataset, file)
                
                # Load and process audio
                y, sr = process_audio(file_path)

                # Save original file
                output_file = os.path.join(OUTPUT_PATH, f"{dataset}_{file}")
                sf.write(output_file, y, sr)
                metadata.append([file, dataset, emotion, output_file])

                # Apply Data Augmentation
                noisy_y = add_noise(y)
                shifted_y = shift_audio(y)

                # Save augmented versions
                sf.write(os.path.join(AUGMENTED_PATH, f"noisy_{dataset}_{file}"), noisy_y, sr)
                sf.write(os.path.join(AUGMENTED_PATH, f"shifted_{dataset}_{file}"), shifted_y, sr)
                metadata.append([f"noisy_{file}", dataset, emotion, output_file])
                metadata.append([f"shifted_{file}", dataset, emotion, output_file])

print("Data preprocessing completed!")

# Save metadata to CSV
df = pd.DataFrame(metadata, columns=["Filename", "Dataset", "Emotion", "Filepath"])
df.to_csv(os.path.join(OUTPUT_PATH, "metadata.csv"), index=False)
print("Metadata saved!")
