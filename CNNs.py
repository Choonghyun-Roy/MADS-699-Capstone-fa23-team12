import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.applications import VGG16
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical

df_cnn = pd.read_csv('tracks_with_genre_small.csv')  

label_encoder = LabelEncoder()
df_cnn['track_genre_top'] = label_encoder.fit_transform(df_cnn['track_genre_top'])

def load_and_convert_to_spectrogram(filename, n_mels=128, duration=30):
    y, sr = librosa.load(filename, duration=duration)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels)
    mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
    return mel_spec_db


df_cnn['spectrogram'] = df_cnn['filename'].apply(lambda x: load_and_convert_to_spectrogram(x))

X_train, X_test, y_train, y_test = train_test_split(
    df_cnn['spectrogram'].values, df_cnn['track_genre_top'].values, test_size=0.2, random_state=42
)

X_train = np.array([x.reshape((X_train.shape[1], X_train.shape[2], 1)) for x in X_train])
X_test = np.array([x.reshape((X_test.shape[1], X_test.shape[2], 1)) for x in X_test])

y_train_encoded = to_categorical(y_train)
y_test_encoded = to_categorical(y_test)

base_model = VGG16(weights='imagenet', include_top=False, input_shape=(X_train.shape[1], X_train.shape[2], 3))

model = Sequential()
model.add(base_model)
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(label_encoder.classes_), activation='softmax'))

for layer in base_model.layers:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer=optimizers.Adam(lr=1e-4), metrics=['accuracy'])

datagen = ImageDataGenerator(
    rotation_range=10,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

model.fit(
    datagen.flow(X_train, y_train_encoded, batch_size=32),
    steps_per_epoch=len(X_train) / 32,
    epochs=20,
    validation_data=(X_test, y_test_encoded)
)

accuracy = model.evaluate(X_test, y_test_encoded)[1]
print(f"Test Accuracy: {accuracy}")
