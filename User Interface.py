import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

user_song_interactions = pd.read_csv('user_song_interactions.csv')

reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(user_song_interactions, reader)
trainset, _ = train_test_split(data, test_size=0)
model = SVD()
model.fit(trainset)

def recommend_songs(user_id, model, user_song_interactions, n_recommendations=10):
    user_history = user_song_interactions.loc[user_id]
    unrated_songs = [song for song in user_song_interactions.columns if user_history[song] == 0]
    user_ratings = [(user_id, song, model.predict(user_id, song).est) for song in unrated_songs]
    top_n_recommendations = sorted(user_ratings, key=lambda x: x[2], reverse=True)[:n_recommendations]
    return [song for _, song, _ in top_n_recommendations]

while True:
    print("Music Recommendation System")
    user_id = int(input("Enter your user ID (or 0 to exit): "))

    if user_id == 0:
        break

    if user_id not in user_song_interactions.index:
        print("User not found. Please enter a valid user ID.")
        continue

    recommended_songs = recommend_songs(user_id, model, user_song_interactions)

    if recommended_songs:
        print("\nRecommended Songs:")
        for i, song in enumerate(recommended_songs, 1):
            print(f"{i}. {song}")
    else:
        print("No recommendations available for this user.")

    print("\n")

print("Goodbye!")
