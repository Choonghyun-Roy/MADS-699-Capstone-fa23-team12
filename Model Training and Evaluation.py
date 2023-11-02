from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy

reader = Reader(rating_scale=(0, 1))
data = Dataset.load_from_df(user_song_interactions, reader)

trainset, testset = train_test_split(data, test_size=0.2)

model = SVD()
model.fit(trainset)

predictions = model.test(testset)

rmse = accuracy.rmse(predictions)
print(f'RMSE: {rmse}')

user_id = 42
n_recommendations = 10

unrated_songs = [song for song in songs if song not in user_song_interactions[user_id].keys()]

user_ratings = [(user_id, song, model.predict(user_id, song).est) for song in unrated_songs]

top_n_recommendations = sorted(user_ratings, key=lambda x: x[2], reverse=True)[:n_recommendations]

print([song for _, song, _ in top_n_recommendations])
