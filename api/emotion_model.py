import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import librosa

# Load your pre-trained model
model = load_model('path/to/your/model.h5')

def extract_features(file_path):
    # Function to extract features from audio file
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    return np.array([mfccs])

def predict_emotion(file_path):
    features = extract_features(file_path)
    prediction = model.predict(features)
    # Assuming your model outputs probabilities for each emotion
    predicted_emotion = np.argmax(prediction)
    return predicted_emotion
