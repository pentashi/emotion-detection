# scripts/predict.py
import os
import numpy as np
import librosa
from keras.models import load_model
import pickle

# Path to the trained model
MODEL_PATH = "../models/emotion_model.h5"
# Path to the processed data (for loading the label encoder)
DATA_PATH = "../data/processed"

# Function to extract MFCC features from an audio file
def extract_features(file_path, max_pad_len=174):
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        pad_width = max_pad_len - mfccs.shape[1]
        mfccs = np.pad(mfccs, pad_width=((0, 0), (0, pad_width)), mode='constant')
        return mfccs
    except Exception as e:
        print(f"Error processing {file_path}: {e}")
        return None

# Function to predict the emotion of an audio file
def predict_emotion(model, file_path, label_encoder):
    features = extract_features(file_path)
    if features is None:
        return None
    features = features[np.newaxis, ..., np.newaxis]
    prediction = model.predict(features)
    predicted_label = label_encoder.inverse_transform([np.argmax(prediction)])
    return predicted_label[0]

if __name__ == "__main__":
    # Load the trained model
    model = load_model(MODEL_PATH)

    # Load the label encoder
    with open(os.path.join(DATA_PATH, 'label_encoder.pkl'), 'rb') as f:
        label_encoder = pickle.load(f)

    # Path to a test audio file
    test_file_path = "../data/test/happy/test.wav"
    # Predict the emotion of the test audio file
    emotion = predict_emotion(model, test_file_path, label_encoder)
    print(f"Predicted emotion: {emotion}")
