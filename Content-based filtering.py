import pandas as pd
import numpy as np
import librosa
from sklearn.metrics.pairwise import cosine_similarity

tracks = pd.read_csv('fma_metadata/tracks.csv')
features = pd.read_csv('audio_features.csv')

def calculate_similarity(user_song_features, dataset_features):
    similarities = cosine_similarity([user_song_features], dataset_features)
    return similarities[0]

user_song_file = 'user_song.mp3'
user_song_features = extract_features(user_song_file)

similarities = calculate_similarity(user_song_features, features)

N = 10
top_n_indices = np.argsort(similarities)[-N:][::-1]

recommended_songs = tracks.loc[top_n_indices]
print(recommended_songs[['track_title', 'artist_name', 'genre']])
