import librosa as lb
import numpy as np
import pandas as pd
import os

FEATURE_OUTPUT_HOME = 'user_uploaded_music'

def save_to_csv(data, columns, output_filename):
    df = pd.DataFrame([data], columns=columns)
    df.to_csv(f"{output_filename}", mode='a', header=False, index=False)

def extract_feature(y, sr, feature_func, *args, **kwargs):
    return feature_func(y=y, sr=sr, *args, **kwargs)

def extract_zero_crossings(y, sr, filename, output_filename):
    zero_crossings = np.sum(lb.zero_crossings(y))
    save_to_csv([filename, zero_crossings], ['track_id', 'zero_crossings'], output_filename)

def extract_tempo(y, sr, filename, output_filename):
    onset_env = lb.onset.onset_strength(y=y, sr=sr)
    tempo, _ = lb.beat.beat_track(onset_envelope=onset_env, sr=sr)
    save_to_csv([filename, tempo], ['track_id', 'tempo'], output_filename)

def extract_spectral_centroid(y, sr, filename, output_filename):
    spectral_centroid = extract_feature(y, sr, lb.feature.spectral_centroid)
    save_to_csv([filename, np.mean(spectral_centroid)], ['track_id', 'spectral_centroid'], output_filename)

def extract_spectral_rolloff(y, sr, filename, output_filename):
    spectral_rolloff = extract_feature(y, sr, lb.feature.spectral_rolloff)
    save_to_csv([filename, np.mean(spectral_rolloff)], ['track_id', 'spectral_rolloff'], output_filename)

def extract_chroma_stft(y, sr, filename, output_filename):
    chroma_stft = extract_feature(y, sr, lb.feature.chroma_stft)
    data = [filename] + list(np.mean(chroma_stft, axis=1))
    save_to_csv(data, ['track_id'] + [f'chroma_stft_{i}' for i in range(1, 13)], output_filename)

def extract_mfccs(y, sr, filename, output_filename):
    mfccs = extract_feature(y, sr, lb.feature.mfcc, n_mfcc=20)
    data = [filename] + list(np.mean(mfccs, axis=1))
    save_to_csv(data, ['track_id'] + [f'MFCC_{i}' for i in range(1, 21)], output_filename)

def extract_harmony_percussive(y, sr, filename, output_filename):
    y_harmonic, y_percussive = lb.effects.hpss(y)
    rms_harmonic = np.mean(lb.feature.rms(y=y_harmonic))
    rms_percussive = np.mean(lb.feature.rms(y=y_percussive))
    save_to_csv([filename, rms_harmonic, rms_percussive], ['track_id', 'rms_harmonic', 'rms_percussive'], output_filename)


def extract_features(file_full_path):
    
    parts = file_full_path.rsplit("/", 1)
    path = parts[0]
    filename = parts[1]
    
# Create CSV headers
    headers = {
        'zero_crossings.csv': ['track_id', 'zero_crossings'],
        'tempo.csv': ['track_id', 'tempo'],
        'spectral_centroid.csv': ['track_id', 'spectral_centroid'],
        'spectral_rolloff.csv': ['track_id', 'spectral_rolloff'],
        'chroma_stft.csv': ['track_id'] + [f'chroma_stft_{i}' for i in range(1, 13)],
        'mfccs.csv': ['track_id'] + [f'MFCC_{i}' for i in range(1, 21)],
        'hpss.csv': ['track_id', 'rms_harmonic', 'rms_percussive']
    }

    for key, value in headers.items():
        pd.DataFrame(columns=value).to_csv(f"{path}/{key}", index=False)
    
    y, sr = lb.load(file_full_path) 
    extract_zero_crossings(y, sr, filename, f"{path}/zero_crossings.csv")
    extract_tempo(y, sr, filename, f"{path}/tempo.csv")
    extract_spectral_centroid(y, sr, filename, f"{path}/spectral_centroid.csv")
    extract_spectral_rolloff(y, sr, filename, f"{path}/spectral_rolloff.csv")
    extract_chroma_stft(y, sr, filename, f"{path}/chroma_stft.csv")
    extract_mfccs(y, sr, filename, f"{path}/mfccs.csv")  
    extract_harmony_percussive(y, sr, filename, f"{path}/hpss.csv") 
    
    ## merge features into a single file
    
    file_names = ["tempo", "hpss", "spectral_centroid", "spectral_rolloff", "zero_crossings", "chroma_stft", "mfccs"]

    # Using a list comprehension to read all dataframes into a list
    dfs = [pd.read_csv(f"{path}/{file_name}.csv") for file_name in file_names]

    merged_df = dfs[0]
    for df in dfs[1:]:
        merged_df = merged_df.merge(df, on='track_id', how='outer')
        
    merged_df.to_csv(f"{path}/{filename}_features.csv")
    
    for file_name in file_names:
        os.remove(f"{path}/{file_name}.csv") 
       
    return merged_df