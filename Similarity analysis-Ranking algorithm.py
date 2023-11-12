import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

features = pd.read_csv('../datasets/fma_metadata/tracks.csv')

def compute_similarity(track1, track2):
    features1 = features[features['track_id'] == track1].iloc[:, 1:].values.flatten()
    features2 = features[features['track_id'] == track2].iloc[:, 1:].values.flatten()
    return cosine_similarity([features1], [features2])[0][0]

# Example
track1_id = '000002'
track2_id = '000005'

similarity = compute_similarity(track1_id, track2_id)
print(f"Similarity between Track {track1_id} and Track {track2_id}: {similarity}")

def recommend_similar_tracks(target_track, num_recommendations=5):
    similarities = []
    for track_id in features['track_id'].unique():
        if track_id != target_track:
            similarity = compute_similarity(target_track, track_id)
            similarities.append((track_id, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    recommendations = similarities[:num_recommendations]
    return recommendations

target_track_id = '000002'
recommendations = recommend_similar_tracks(target_track_id, num_recommendations=5)

print(f"Recommendations for Track {target_track_id}:")
for i, (track_id, similarity) in enumerate(recommendations, 1):
    print(f"{i}. Track {track_id} (Similarity: {similarity})")


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
