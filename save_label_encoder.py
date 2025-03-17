import pickle
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

# Paths
FEATURES_PATH = r"C:\Users\ASUS\Desktop\mcode\feature_extraction"
LABEL_ENCODER_PATH = os.path.join(FEATURES_PATH, "label_encoder.pkl")

# Load extracted features
FEATURES_FILE = os.path.join(FEATURES_PATH, "features.pkl")

with open(FEATURES_FILE, "rb") as f:
    data = pickle.load(f)

y = np.array([entry[1] for entry in data])  # Extract emotion labels

# Create Label Encoder
label_encoder = LabelEncoder()
label_encoder.fit(y)

# Save label encoder
with open(LABEL_ENCODER_PATH, "wb") as f:
    pickle.dump(label_encoder, f)

print(f"Label encoder saved at: {LABEL_ENCODER_PATH}")
