# scripts/train_model.py
import os
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, Conv2D, MaxPooling2D
from keras.utils import to_categorical

# Path to the processed data
DATA_PATH = "../data/processed"
# Path to save the trained model
MODEL_PATH = "../models/emotion_model.h5"

# Function to load the preprocessed data
def load_data(data_path):
    with open(os.path.join(data_path, 'X_train.pkl'), 'rb') as f:
        X_train = pickle.load(f)
    with open(os.path.join(data_path, 'X_test.pkl'), 'rb') as f:
        X_test = pickle.load(f)
    with open(os.path.join(data_path, 'y_train.pkl'), 'rb') as f:
        y_train = pickle.load(f)
    with open(os.path.join(data_path, 'y_test.pkl'), 'rb') as f:
        y_test = pickle.load(f)
    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    # Load the preprocessed data
    X_train, X_test, y_train, y_test = load_data(DATA_PATH)

    # Add a channel dimension to the data
    X_train = X_train[..., np.newaxis]
    X_test = X_test[..., np.newaxis]

    # Encode the labels as integers
    label_encoder = LabelEncoder()
    y_train = label_encoder.fit_transform(y_train)
    y_test = label_encoder.transform(y_test)

    # Convert the labels to one-hot encoded format
    y_train = to_categorical(y_train)
    y_test = to_categorical(y_test)

    # Define the CNN model architecture
    model = Sequential()
    model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(40, 174, 1)))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D((2, 2)))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(len(np.unique(y_train)), activation='softmax'))

    # Compile the model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    # Train the model
    model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test))

    # Save the trained model
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    model.save(MODEL_PATH)

    # Save the label encoder for later use
    with open(os.path.join(DATA_PATH, 'label_encoder.pkl'), 'wb') as f:
        pickle.dump(label_encoder, f)

    print("Model training completed and saved.")
