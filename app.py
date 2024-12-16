import os
import librosa
import numpy as np
import pandas as pd
from flask import Flask, request, render_template, jsonify
from werkzeug.utils import secure_filename
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical
import joblib
import kagglehub
import shutil
from pathlib import Path

# --- Flask App Setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# --- Helper Functions ---

def extract_features(file_path, duration=30):
    """Extract MFCC features from an audio file."""
    y, sr = librosa.load(file_path, duration=duration)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    mfcc_mean = np.mean(mfcc, axis=1)
    return mfcc_mean

def apply_emotion_filter(file_path, emotion):
    """Apply emotion-based transformations to the audio file."""
    y, sr = librosa.load(file_path)
    if emotion == 'happy':
        y = librosa.effects.pitch_shift(y, sr, n_steps=2)  # Increase pitch
    elif emotion == 'calm':
        y = librosa.effects.time_stretch(y, rate=0.8)  # Slow down
    elif emotion == 'energetic':
        y = librosa.effects.time_stretch(y, rate=1.2)  # Speed up
    elif emotion == 'sad':
        y = librosa.effects.pitch_shift(y, sr, n_steps=-2)  # Decrease pitch
    output_file = os.path.join(app.config['UPLOAD_FOLDER'], f"edited_{emotion}.wav")
    librosa.output.write_wav(output_file, y, sr)
    return output_file

#TODO
def prepare_dataset(data_dir="./data/gtzan", kaggle_dataset="andradaolteanu/gtzan-dataset-music-genre-classification", download=True):
    """
    Download and prepare the GTZAN dataset for music genre classification.
    """
    dataset_dir = Path(data_dir) / "genres_original"
    if not download and dataset_dir.exists():
        print(f"[INFO] GTZAN dataset already exists at {dataset_dir}.")
        return dataset_dir

    print(f"[INFO] Preparing GTZAN dataset at {data_dir}...")

    try:
        # Download the dataset using KaggleHub
        print(f"[INFO] Downloading {kaggle_dataset}...")
        path = kagglehub.dataset_download(kaggle_dataset)
        print(f"[INFO] Dataset downloaded to cache at {path}.")

        # Locate the correct path for `genres_original`
        genres_path = Path(path)
        if not genres_path.exists():
            raise FileNotFoundError(f"[ERROR] 'genres_original' folder not found in {genres_path}. Check the dataset structure.")

        # Move the `genres_original` folder to the local directory
        for genre_dir in genres_path.glob("*"):
            if genre_dir.is_dir():
                print(f"[INFO] Moving files from {genre_dir} to {dataset_dir}...")
                dataset_dir.mkdir(parents=True, exist_ok=True)
                for file in genre_dir.glob("*"):
                    shutil.copy(file, dataset_dir / file.name)
                print(f"[INFO] Moved files for genre: {genre_dir.name}")

        print(f"[INFO] Dataset successfully prepared in {dataset_dir}.")
        return dataset_dir

    except Exception as e:
        print(f"[ERROR] Failed to download or prepare the GTZAN dataset: {str(e)}")
        raise


def train_model(df, genres):
    """Train a neural network for music genre classification."""
    X = np.array(df['features'].tolist())
    y = np.array(df['genre'].tolist())

    encoder = LabelEncoder()
    y_encoded = encoder.fit_transform(y)
    y_categorical = to_categorical(y_encoded)

    X_train, X_test, y_train, y_test = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, 'scaler.pkl')

    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train_scaled.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(len(genres), activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.fit(X_train_scaled, y_train, validation_data=(X_test_scaled, y_test), epochs=30, batch_size=32)

    model.save('genre_classification_model.h5')
    return model, scaler, encoder

# --- Flask Routes ---
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    """Handle file upload and predict genre."""
    if 'file' not in request.files:
        return "No file uploaded.", 400
    file = request.files['file']
    if file.filename == '':
        return "No selected file.", 400

    filename = secure_filename(file.filename)
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    features = extract_features(file_path)
    features_scaled = scaler.transform([features])
    genre_prediction = model.predict(features_scaled)
    predicted_genre = encoder.inverse_transform([np.argmax(genre_prediction)])[0]

    return jsonify({"genre": predicted_genre, "file_path": file_path})

@app.route('/edit', methods=['POST'])
def edit():
    """Apply emotion-based editing to an uploaded audio file."""
    data = request.get_json()
    file_path = data.get('file_path')
    emotion = data.get('emotion')

    if not file_path or not emotion:
        return "File path or emotion missing.", 400

    edited_file = apply_emotion_filter(file_path, emotion)
    return jsonify({"edited_file": edited_file})

# --- Main Execution ---
if __name__ == '__main__':
    dataset_dir = './data/gtzan'
    if not os.path.exists('genre_classification_model.h5'):
        print("Preparing dataset and training model...")
        df, genres = prepare_dataset(data_dir=dataset_dir)
        model, scaler, encoder = train_model(df, genres)
    else:
        print("Loading pre-trained model and scaler...")
        model = load_model('genre_classification_model.h5')
        scaler = joblib.load('scaler.pkl')
        genres = ['blues', 'classical', 'country', 'disco', 'hiphop', 
                  'jazz', 'metal', 'pop', 'reggae', 'rock']
        encoder = LabelEncoder()
        encoder.fit(genres)

    app.run(debug=True)
