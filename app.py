import streamlit as st
import numpy as np
import librosa
import librosa.display
import soundfile as sf
import tensorflow as tf
import os
from pydub import AudioSegment
from sklearn.preprocessing import LabelEncoder
import pickle
from streamlit_webrtc import webrtc_streamer, WebRtcMode, AudioProcessorBase
import av
import tempfile

# Paths
MODEL_PATH = "C:\\Users\\ASUS\\Desktop\\mcode\\models"
FEATURES_PATH = "C:\\Users\\ASUS\\Desktop\\mcode\\feature_extraction"
LABEL_ENCODER_PATH = os.path.join(FEATURES_PATH, "label_encoder.pkl")

# Load Models
@st.cache_resource
def load_models():
    mlp_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, "MLP.h5"))
    cnn_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, "CNN.h5"))
    cnn_bilstm_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, "CNN_BiLSTM.h5"))
    return mlp_model, cnn_model, cnn_bilstm_model

mlp_model, cnn_model, cnn_bilstm_model = load_models()

# Load Label Encoder
with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)

# Function to extract features
def extract_features(audio_path, expected_shape=155):
    y, sr = librosa.load(audio_path, sr=22050)
    features = {
        "mfcc": librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20).mean(axis=1),
        "mel_spectrogram": librosa.feature.melspectrogram(y=y, sr=sr).mean(axis=1),
        "zcr": librosa.feature.zero_crossing_rate(y).mean(),
        "rms": librosa.feature.rms(y=y).mean(),
        "chroma": librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1)
    }
    
    feature_vector = np.hstack(list(features.values()))
    
    # Fix shape mismatch (truncate or pad)
    if feature_vector.shape[0] > expected_shape:
        feature_vector = feature_vector[:expected_shape]
    elif feature_vector.shape[0] < expected_shape:
        feature_vector = np.pad(feature_vector, (0, expected_shape - feature_vector.shape[0]))

    return feature_vector

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_path, wav_path):
    audio = AudioSegment.from_mp3(mp3_path)
    audio.export(wav_path, format="wav")
    return wav_path

# Streamlit UI
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload an audio file (`.wav` or `.mp3`) or record an audio to predict its emotion.")

# Reset Button
def reset_app():
    st.experimental_rerun()

if st.button("🔄 Reset"):
    reset_app()

uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"])

# Record Audio Button using streamlit-webrtc
st.subheader("🎤 Record Audio")
recorded_audio = None

class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.recorded_audio = None

    def recv(self, frame: av.AudioFrame) -> av.AudioFrame:
        self.recorded_audio = frame.to_ndarray()
        return frame

webrtc_ctx = webrtc_streamer(
    key="speech-recorder",
    mode=WebRtcMode.SENDRECV,
    audio_processor_factory=AudioProcessor,
    video_processor_factory=None,  # Disable video
    media_stream_constraints={"video": False, "audio": True},
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},  # Use Google's STUN server
)


if webrtc_ctx and webrtc_ctx.audio_processor:
    audio_processor = webrtc_ctx.audio_processor
    if audio_processor.recorded_audio is not None:
        temp_audio_path = tempfile.NamedTemporaryFile(delete=False, suffix=".wav").name
        sf.write(temp_audio_path, audio_processor.recorded_audio, 22050)
        st.audio(temp_audio_path, format="audio/wav")
        uploaded_file = temp_audio_path

if uploaded_file is not None:
    file_path = os.path.join("temp_audio", uploaded_file.name) if isinstance(uploaded_file, str) else "temp_audio/uploaded_audio.wav"
    os.makedirs("temp_audio", exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read() if not isinstance(uploaded_file, str) else open(uploaded_file, "rb").read())
    
    if file_path.endswith(".mp3"):
        wav_path = file_path.replace(".mp3", ".wav")
        file_path = convert_mp3_to_wav(file_path, wav_path)
    
    st.audio(file_path, format='audio/wav')
    
    features = extract_features(file_path).reshape(1, -1)
    features_cnn = features.reshape(features.shape[0], features.shape[1], 1)
    
    # Predict using models
    predictions = {
        "MLP": mlp_model.predict(features)[0],
        "CNN": cnn_model.predict(features_cnn)[0],
        "CNN + BiLSTM": cnn_bilstm_model.predict(features_cnn)[0]
    }
    
    st.subheader("Predicted Emotions")
    for model_name, prediction in predictions.items():
        emotion = label_encoder.inverse_transform([np.argmax(prediction)])[0]
        st.write(f"**{model_name}:** {emotion}")