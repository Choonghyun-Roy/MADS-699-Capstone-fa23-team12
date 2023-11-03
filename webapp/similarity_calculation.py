import pandas as pd
import numpy as np
from scipy.spatial import distance
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

META_FILE = '../preprocessing/datasets/tracks_with_genre_small.csv'
FEATURE_FILE = '../feature_extraction/features/all_features.csv'
LABEL = 'depth_1_genre_name'

def get_current_music_data():
    # load data and merge features and label
    df_meta = pd.read_csv(META_FILE)
    df_features = pd.read_csv(FEATURE_FILE)
    merged_df = df_meta.merge(df_features, on='track_id', how='outer')

    # drop unnecessary column and rows including null values)
    
    used_columns = [LABEL] + [col for col in df_features.columns if col != 'Unnamed: 0']
    merged_df = merged_df[used_columns].dropna()
    return merged_df


def weighted_euclidean_distance(df, feature_importance):
    # Apply weights to the features
    weighted_features = df * feature_importance
    
    # Initialize an empty DataFrame for distances
    distances_df = pd.DataFrame(index=weighted_features.index, columns=weighted_features.index)
    
    # Calculate the Euclidean distance for each unique pair of rows
    for i in weighted_features.index:
        for j in weighted_features.index:
            # Fill the DataFrame symmetrically since the distance is a symmetric measure
            if pd.isnull(distances_df.at[i, j]):
                # Compute the distance
                dist = distance.euclidean(df.loc[i], df.loc[j])
                distances_df.at[i, j] = dist
                distances_df.at[j, i] = dist
    
    # Set the diagonal to 0.0 since the distance of an observation to itself is always 0
    pd.fill_diagonal(distances_df.values, 0)
    upper_triangle = distances_df.where(np.triu(np.ones(distances_df.shape), k=1).astype(np.bool))
    
    # Find the minimum and maximum values
    min_distance = upper_triangle.min().min()
    max_distance = upper_triangle.max().max()
    
    # Apply the normalization formula
    normalized_df = 1 - (distances_df - min_distance) / (max_distance - min_distance)
    
    # Fill the diagonal with 1s since the distance to itself should be the closest
    np.fill_diagonal(normalized_df.values, 1)
    
    return normalized_df


def weighted_cosine_similarity(df, feature_importance):
    # Apply weights to the features
    weighted_features = df * feature_importance
    
    # Compute the cosine similarity matrix
    similarity_matrix = cosine_similarity(weighted_features)
    
    # Convert the similarity matrix to a DataFrame
    similarity_df = pd.DataFrame(similarity_matrix, index=df.index, columns=df.index)
    
    return similarity_df
    

def gain_similarity_matrix(track_id, X, genre=None, weighted=False, feature_importance=None):
    current_music_df = get_current_music_data()
    X.insert(loc=0, column=LABEL, value=[genre])
    X.insert(loc=1, column='track_id', value=[track_id])
    combined_data = pd.concat([current_music_df, X], axis=0, ignore_index=True)
    
    # Normalize feature values.
    feature_list = [col for col in combined_data.columns if col not in ['Unnamed: 0', 'track_id', LABEL]]
    scaler = MinMaxScaler()
    combined_data[feature_list] = scaler.fit_transform(combined_data[feature_list])

    combined_data = combined_data.set_index('track_id')
   
    if genre is not None:
        combined_data = combined_data[combined_data[LABEL] == genre]
    
    # get similarity
    if not weighted:
        similarity = cosine_similarity(combined_data[feature_list])
        print('cosine similarity')
        print(similarity)
    else:
        similarity = weighted_cosine_similarity(combined_data[feature_list], feature_importance)
        print('weighted cosine similarity')
        print(similarity)
        
    labels = combined_data[LABEL]
    result = pd.DataFrame(similarity, index=labels.index, columns=labels.index)
    return result


def get_similar_music(track_id, X, genre, n=10, weighted=False, feature_importance=None):
    df_meta = pd.read_csv(META_FILE)

    # Get the similarity scores
    series = gain_similarity_matrix(track_id, X, genre, weighted, feature_importance)[track_id]
  
    # If it's a DataFrame, drop duplicates and select the track_id column to get a Series
    if isinstance(series, pd.DataFrame):
        series = series.drop_duplicates(subset=track_id, keep='first')[track_id].iloc[:, 0]
    
     # Sort the Series
    series = series.sort_values(ascending=False)
    series = series.drop(track_id)
    
    # Convert to DataFrame
    sorted_df = series.reset_index()
    sorted_df.columns = ['track_id', 'similarity_score']
    
    # Remove duplicates based on track_id from sorted_df
    sorted_df = sorted_df.drop_duplicates(subset='track_id', keep='first')
    
    # Select the top n songs after removing duplicates
    top_n = sorted_df.head(n)
    
    # Ensure df_meta also doesn't have duplicates for the same track_id
    df_meta_unique = df_meta.drop_duplicates(subset='track_id', keep='first')
    
    # Join with df_meta_unique to get the genre of the recommended songs
    result = top_n.merge(df_meta_unique[['track_id', LABEL, 'artist_name', 'track_title']], on='track_id', how='left')
    
    return result