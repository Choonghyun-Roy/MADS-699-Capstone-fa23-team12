import pandas as pd

FEATURES_VALID = 'preprocessing/datasets/ohe_25K_tracks_features_and_labels_for_validation.csv'

def extract_features(track_id):
    
    df = pd.read_csv(FEATURES_VALID)
    used_columns = [col for col in df.columns if col not in ['Unnamed: 0', 'track_genre_top', 'track_title', 'artist_name', 'set_split', 'set_subset']]
    df = df[used_columns]
    filtered_df = df[df['track_id'] == track_id]
    
    return filtered_df