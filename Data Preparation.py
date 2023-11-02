import pandas as pd

tracks = pd.read_csv('fma_metadata/tracks.csv')

import librosa
import librosa.display
import numpy as np

def extract_features(audio_file):
    y, sr = librosa.load(audio_file)
    mfccs = librosa.feature.mfcc(y=y, sr=sr)
    chroma = librosa.feature.chroma_stft(y=y, sr=sr)
    tempo = librosa.beat.tempo(y=y, sr=sr)
    spectral_contrast = librosa.feature.spectral_contrast(y=y, sr=sr)

    features = np.concatenate((mfccs, chroma, [tempo], spectral_contrast), axis=None)

    return features

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()
tracks['genre_encoded'] = label_encoder.fit_transform(tracks['genre'])

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

# Create a user-song interaction matrix
user_song_matrix = create_user_song_matrix(user_interactions)

# Create a feature matrix
feature_matrix = create_feature_matrix(features)
