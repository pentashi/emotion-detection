# scripts/preprocess.py
import os
import librosa
import numpy as np
import pickle
from sklearn.model_selection import train_test_split

# Path to the dataset root directory
DATA_PATH = "../data"
# Path to save the processed data
OUTPUT_PATH = "../data/processed"

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

# Function to load and process the dataset
def load_data(data_path):
    # List of emotion labels (subdirectory names)
    emotions = ['angry', 'happy', 'neutral', 'sad']
    X, y = [], []
    for emotion in emotions:
        emotion_path = os.path.join(data_path, emotion)
        for file_name in os.listdir(emotion_path):
            file_path = os.path.join(emotion_path, file_name)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(emotion)
    return np.array(X), np.array(y)

if __name__ == "__main__":
    # Load and preprocess the data
    X, y = load_data(DATA_PATH)
    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Create the output directory if it doesn't exist
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    # Save the preprocessed data as pickle files
    with open(os.path.join(OUTPUT_PATH, 'X_train.pkl'), 'wb') as f:
        pickle.dump(X_train, f)
    with open(os.path.join(OUTPUT_PATH, 'X_test.pkl'), 'wb') as f:
        pickle.dump(X_test, f)
    with open(os.path.join(OUTPUT_PATH, 'y_train.pkl'), 'wb') as f:
        pickle.dump(y_train, f)
    with open(os.path.join(OUTPUT_PATH, 'y_test.pkl'), 'wb') as f:
        pickle.dump(y_test, f)
    print("Data preprocessing completed.")
