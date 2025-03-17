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
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration, AudioProcessorBase
import av
import tempfile
import time
import queue
import threading

# App state management
if 'audio_file_path' not in st.session_state:
    st.session_state.audio_file_path = None
if 'prediction_ready' not in st.session_state:
    st.session_state.prediction_ready = False
if 'reset_requested' not in st.session_state:
    st.session_state.reset_requested = False
if 'audio_recorded' not in st.session_state:
    st.session_state.audio_recorded = False

# Paths
MODEL_PATH = "models"  # Update with your relative path
FEATURES_PATH = "feature_extraction"  # Update with your relative path
LABEL_ENCODER_PATH = os.path.join(FEATURES_PATH, "label_encoder.pkl")

# Load Models
@st.cache_resource
def load_models():
    try:
        mlp_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, "MLP.h5"))
        cnn_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, "CNN.h5"))
        cnn_bilstm_model = tf.keras.models.load_model(os.path.join(MODEL_PATH, "CNN_BiLSTM.h5"))
        return mlp_model, cnn_model, cnn_bilstm_model
    except Exception as e:
        st.error(f"Error loading models: {str(e)}")
        return None, None, None

# Load Label Encoder
@st.cache_resource
def load_label_encoder():
    try:
        with open(LABEL_ENCODER_PATH, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        st.error(f"Error loading label encoder: {str(e)}")
        return None

# Function to extract features
def extract_features(audio_path, expected_shape=155):
    try:
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
    except Exception as e:
        st.error(f"Error extracting features: {str(e)}")
        return None

# Convert MP3 to WAV
def convert_mp3_to_wav(mp3_path, wav_path):
    try:
        audio = AudioSegment.from_mp3(mp3_path)
        audio.export(wav_path, format="wav")
        return wav_path
    except Exception as e:
        st.error(f"Error converting MP3 to WAV: {str(e)}")
        return None

# Predict emotion
def predict_emotion(file_path):
    try:
        # Ensure models are loaded
        mlp_model, cnn_model, cnn_bilstm_model = load_models()
        label_encoder = load_label_encoder()
        
        if not all([mlp_model, cnn_model, cnn_bilstm_model, label_encoder]):
            st.error("Models or label encoder not loaded properly")
            return None
        
        # Extract features
        features = extract_features(file_path)
        if features is None:
            return None
            
        features = features.reshape(1, -1)
        features_cnn = features.reshape(features.shape[0], features.shape[1], 1)
        
        # Predict using models
        predictions = {
            "MLP": mlp_model.predict(features, verbose=0)[0],
            "CNN": cnn_model.predict(features_cnn, verbose=0)[0],
            "CNN + BiLSTM": cnn_bilstm_model.predict(features_cnn, verbose=0)[0]
        }
        
        results = {}
        for model_name, prediction in predictions.items():
            emotion = label_encoder.inverse_transform([np.argmax(prediction)])[0]
            confidence = float(np.max(prediction)) * 100
            results[model_name] = {"emotion": emotion, "confidence": confidence}
        
        return results
    except Exception as e:
        st.error(f"Error predicting emotion: {str(e)}")
        return None

# Streamlit UI
st.title("🎙️ Speech Emotion Recognition")
st.write("Upload an audio file (`.wav` or `.mp3`) or record an audio to predict its emotion.")

# Reset Button
if st.button("🔄 Reset"):
    st.session_state.audio_file_path = None
    st.session_state.prediction_ready = False
    st.session_state.audio_recorded = False
    st.session_state.reset_requested = True
    st.experimental_rerun()

# File uploader
uploaded_file = st.file_uploader("Upload Audio File", type=["wav", "mp3"], key="file_uploader")

# If file is uploaded
if uploaded_file is not None:
    # Create temp directory if it doesn't exist
    os.makedirs("temp_audio", exist_ok=True)
    
    # Save uploaded file
    file_path = os.path.join("temp_audio", uploaded_file.name)
    with open(file_path, "wb") as f:
        f.write(uploaded_file.read())
    
    # Convert MP3 to WAV if needed
    if file_path.endswith(".mp3"):
        wav_path = file_path.replace(".mp3", ".wav")
        file_path = convert_mp3_to_wav(file_path, wav_path)
    
    st.session_state.audio_file_path = file_path
    st.audio(file_path, format='audio/wav')

# Record Audio Button using streamlit-webrtc
st.subheader("🎤 Record Audio")

# Define audio processor with queue and threading for smoother recording
class AudioProcessor(AudioProcessorBase):
    def __init__(self):
        self.audio_queue = queue.Queue()
        self.audio_frames = []
        self.recording = True
        self.thread = threading.Thread(target=self._recording_thread)
        self.thread.daemon = True
        self.thread.start()
    
    def _recording_thread(self):
        while self.recording:
            try:
                frame = self.audio_queue.get(timeout=1)
                self.audio_frames.append(frame.to_ndarray())
            except queue.Empty:
                continue
            except Exception as e:
                st.error(f"Error in recording thread: {str(e)}")
                break
    
    def recv_queued(self, frames):
        try:
            for frame in frames:
                self.audio_queue.put(frame)
            return frames
        except Exception as e:
            st.error(f"Error in recv_queued: {str(e)}")
            return frames
    
    def save_audio(self):
        try:
            if not self.audio_frames:
                return None
                
            # Combine all audio frames
            audio_data = np.concatenate(self.audio_frames, axis=0)
            
            # Save to temp file
            temp_audio_path = os.path.join("temp_audio", f"recorded_audio_{int(time.time())}.wav")
            os.makedirs("temp_audio", exist_ok=True)
            sf.write(temp_audio_path, audio_data, 22050)
            
            return temp_audio_path
        except Exception as e:
            st.error(f"Error saving audio: {str(e)}")
            return None

# Configure WebRTC
rtc_configuration = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# Create audio processor only if we're recording
audio_processor = None
webrtc_ctx = webrtc_streamer(
    key="speech-recorder",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=rtc_configuration,
    audio_processor_factory=AudioProcessor,
    video_processor_factory=None,
    media_stream_constraints={"video": False, "audio": True},
    async_processing=True,
)

# Handle recording
if webrtc_ctx.state.playing:
    st.info("Recording in progress...")
    audio_processor = webrtc_ctx.audio_processor
    
    # Stop recording button
    if st.button("Stop Recording"):
        webrtc_ctx.video_processor = None
        if audio_processor:
            recorded_file_path = audio_processor.save_audio()
            if recorded_file_path:
                st.session_state.audio_file_path = recorded_file_path
                st.session_state.audio_recorded = True
                st.experimental_rerun()

# Display recorded audio
if st.session_state.audio_recorded and st.session_state.audio_file_path:
    st.subheader("Recorded Audio")
    st.audio(st.session_state.audio_file_path, format="audio/wav")

# Predict button
if st.session_state.audio_file_path and st.button("Predict Emotion"):
    with st.spinner("Analyzing audio..."):
        results = predict_emotion(st.session_state.audio_file_path)
        
        if results:
            st.session_state.prediction_ready = True
            st.subheader("Predicted Emotions")
            
            # Display predictions
            for model_name, result in results.items():
                emotion = result["emotion"]
                confidence = result["confidence"]
                
                # Create a progress bar for confidence
                st.write(f"**{model_name}:** {emotion} ({confidence:.2f}%)")
                st.progress(confidence / 100)