import pandas as pd

FEATURES_VALID = 'feature_extraction/features/all_features_medium_with_var.csv'

def extract_features(track_id):
    
    df = pd.read_csv(FEATURES_VALID)
    filtered_df = df[df['track_id'] == track_id]
    filtered_df = filtered_df.drop('Unnamed: 0', axis=1)
    
    return filtered_df