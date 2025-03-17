import os
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Conv1D, MaxPooling1D, Flatten, LSTM, Bidirectional, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical

# Paths
FEATURES_PATH = r"C:\Users\ASUS\Desktop\mcode\feature_extraction\features.pkl"
MODEL_OUTPUT = r"C:\Users\ASUS\Desktop\mcode\models"

# Ensure model output directory exists
os.makedirs(MODEL_OUTPUT, exist_ok=True)

# Load extracted features
with open(FEATURES_PATH, "rb") as f:
    data = pickle.load(f)

X = np.array([entry[0] for entry in data])  # Features
y = np.array([entry[1] for entry in data])  # Labels (Emotions)

# Encode emotion labels
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)
y_one_hot = to_categorical(y_encoded)  # Convert labels to one-hot encoding

# Split dataset (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y_one_hot, test_size=0.2, random_state=42)

print(f"Training data shape: {X_train.shape}, Labels: {y_train.shape}")

# Optimizer
optimizer = Adam(learning_rate=0.0005)

# Define MLP Model (Unchanged)
def build_mlp(input_shape, num_classes):
    model = Sequential([
        Dense(256, activation='relu', input_shape=(input_shape,)),
        Dropout(0.2),
        Dense(128, activation='relu'),
        Dropout(0.2),
        Dense(64, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define CNN Model (Unchanged)
def build_cnn(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
        MaxPooling1D(pool_size=2),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Define CNN + BiLSTM Model (Fixed Overfitting + Reduced Epochs)
def build_cnn_bilstm(input_shape, num_classes):
    model = Sequential([
        Conv1D(64, kernel_size=3, activation='relu', input_shape=(input_shape, 1)),
        BatchNormalization(),  # Helps stabilize training
        MaxPooling1D(pool_size=2),
        Dropout(0.3),  # Increased dropout to prevent overfitting
        Bidirectional(LSTM(128, return_sequences=True, dropout=0.2, recurrent_dropout=0.2)),  # LSTM Regularization
        Flatten(),
        Dense(128, activation='relu', kernel_regularizer=l2(0.0001)),  # Reduced L2 regularization
        Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# Early Stopping to prevent overfitting
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Load existing models (MLP & CNN remain unchanged)
mlp_model_path = os.path.join(MODEL_OUTPUT, "MLP.h5")
cnn_model_path = os.path.join(MODEL_OUTPUT, "CNN.h5")

if os.path.exists(mlp_model_path):
    print("Loading existing MLP model...")
    mlp_model = load_model(mlp_model_path)
else:
    print("Training MLP model...")
    mlp_model = build_mlp(X_train.shape[1], y_train.shape[1])
    mlp_model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
    mlp_model.save(mlp_model_path)
    print("MLP training complete! Model saved.")

if os.path.exists(cnn_model_path):
    print("Loading existing CNN model...")
    cnn_model = load_model(cnn_model_path)
else:
    print("Training CNN model...")
    cnn_model = build_cnn(X_train.shape[1], y_train.shape[1])
    X_train_cnn = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_test_cnn = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)
    cnn_model.fit(X_train_cnn, y_train, epochs=50, batch_size=32, validation_data=(X_test_cnn, y_test), callbacks=[early_stopping])
    cnn_model.save(cnn_model_path)
    print("CNN training complete! Model saved.")

# Train Only CNN + BiLSTM Model
print("Training CNN + BiLSTM...")
cnn_bilstm_model = build_cnn_bilstm(X_train.shape[1], y_train.shape[1])

X_train_bilstm = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
X_test_bilstm = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

cnn_bilstm_model.fit(
    X_train_bilstm, y_train, 
    epochs=12, batch_size=32,  # Reduced epochs from 50 → 30
    validation_data=(X_test_bilstm, y_test), 
    callbacks=[early_stopping]
)

cnn_bilstm_model.save(os.path.join(MODEL_OUTPUT, "CNN_BiLSTM.h5"))
print("CNN + BiLSTM training complete! Model saved.")

print("All models are ready!")
