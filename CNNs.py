import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, pairwise_distances

le = LabelEncoder()
genre_labels_encoded = le.fit_transform(genre_labels)

X_train, X_test, y_train, y_test = train_test_split(spectrogram_data, genre_labels_encoded, test_size=0.2, random_state=42)

model = models.Sequential()

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

model.fit(X_train, y_train, epochs=10, batch_size=32, validation_data=(X_test, y_test))

user_song_features = model.predict(X_test)

similarity_scores = 1 - pairwise_distances(user_song_features, model.predict(X_train), metric='cosine')

num_recommendations = 5
top_indices = np.argsort(similarity_scores[0])[-num_recommendations:][::-1]

recommended_genres = le.inverse_transform(y_train[top_indices])

print("Recommended Genres:")
for genre in recommended_genres:
    print(genre)
