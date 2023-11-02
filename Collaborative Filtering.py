import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

user_song_interactions = pd.read_csv('user_song_interactions.csv')

def calculate_user_similarity(user_id, user_song_interactions):
    user_history = user_song_interactions.loc[user_id]

    similarities = cosine_similarity([user_history], user_song_interactions)
    return similarities[0]

user_id = 42

user_similarities = calculate_user_similarity(user_id, user_song_interactions)

N = 5
similar_user_indices = np.argsort(user_similarities)[-N:][::-1]

recommended_songs = set()
for user_index in similar_user_indices:
    similar_user_history = user_song_interactions.iloc[user_index]
    recommended_songs.update(similar_user_history.index[similar_user_history == 1])

user_history = user_song_interactions.loc[user_id]
recommended_songs.difference_update(user_history.index[user_history == 1])

recommended_songs = list(recommended_songs)[:N]
print(recommended_songs)
