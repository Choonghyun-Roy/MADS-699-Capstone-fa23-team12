import streamlit as st
import pandas as pd
from pathlib import Path
from feature_extraction import extract_features
from similarity_calculation import get_similar_music
from pycaret.classification import load_model
from dbconnection import execute_query

# Constants
N_OF_RECOMMENDATIONS = 10  # You can adjust the number of recommendations
UPLOAD_HOME = Path('webapp/user_uploaded_music')  # Directory for user-uploaded files
FILE_PATH = Path('webapp/music_list')  # Directory where music list files are stored
VALID_META_FILE = Path('preprocessing/datasets/ohe_25K_tracks_features_and_labels_for_validation.csv')

SIMPLE_COSINE_SIMILARITY = 'Simple Cosine Similarity'
LIGHTGBM = 'LightGBM'
CNN = 'CNN'
LSTM = 'LSTM'

# Load trained model
supervised_learning_model = load_model('lightgbm_model_25K_88F_231124')
cnn_model = load_model('xgboost_model_25K_74F_231111')
lstm_model = load_model('xgboost_model_25K_74F_231111')
N_OF_RECOMMEND = 10
             
def insert_feedback(user_name, selected_model, org_track_id, track_id, like_yn):
    insert_query = """
        INSERT INTO user_feedback (user_name, model_type, original_track_id, recommended_track_id, like_yn, apply_yn) 
        VALUES (%s, %s, %s, %s, %s, 'N');
    """
    data = (user_name, selected_model, org_track_id, track_id, like_yn)
    execute_query(insert_query, data)

def all_genre_submitted():
    st.session_state.all_genre_submitted = True
    
def all_genre_reset():
    st.session_state.all_genre_submitted = False
    
def selected_genre_submitted():
    st.session_state.selected_genre_submitted = True
    
def selected_genre_reset():
    st.session_state.selected_genre_submitted = False
    
def predict_genre_submitted():
    st.session_state.predict_genre_submitted = True
    
def predict_genre_reset():
    st.session_state.predict_genre_submitted = False
   
def feedback_submitted():
    st.session_state.feedback_submitted = True
    
def feedback_reset():
    st.session_state.feedback_submitted = False
                      
def show_result(org_track_id, selected_model, result):          
    
    # Use session state to store toggle states
    st.session_state.toggle_states = {}
        
    with st.form(key=f'feedback_form_{org_track_id}_{selected_model}'):
        # Display headers
        cols = st.columns([1, 1, 1, 2, 2, 3, 2])
        headers = ['track_id', 'score', 'genre', 'artist', 'track_title', 'play', 'feedback']
        for col, header in zip(cols, headers):
            col.write(header)
        
        # Display each track row with its feedback checkbox
        for index, row in result.iterrows():
            cols = st.columns([1, 1, 1, 2, 2, 3, 2])
            track_id = row['track_id']
            for col, field in zip(cols[:-2], [track_id, round(row['similarity_score'], 4), row['track_genre_top'], row['artist_name'], row['track_title']]):
                col.write(field)
            file_name = f"{FILE_PATH}/{int(track_id):06d}.mp3"
            cols[5].audio(file_name, format='audio/mp3')
            feedback_key = f"like_{track_id}_{org_track_id}_{selected_model}"
            liked = cols[6].checkbox('Like', key=feedback_key, value=st.session_state.toggle_states.get(feedback_key, False))
            st.session_state.toggle_states[feedback_key] = liked  # Update session state
        
        user_name = st.text_input("Enter your username", key=f"user_name_{org_track_id}_{selected_model}")
        st.form_submit_button("Send feedback", on_click=feedback_submitted)

    if 'feedback_submitted' in st.session_state and st.session_state.feedback_submitted: 
        print('feedback_submitted')
        with st.spinner("Processing feedback..."):
            # Process feedback for each track using the session state information
            for track_id, liked in st.session_state.toggle_states.items():
                numeric_track_id = int(track_id.split('_')[1])
                insert_feedback(user_name, selected_model, str(int(org_track_id)).zfill(6), str(int(numeric_track_id)).zfill(6), liked)
            st.success('Thank you!!! Your feedback is successfully submitted!')
        
        # Clear feedback after submission
        for key in st.session_state.toggle_states.keys():
            st.session_state.toggle_states[key] = False    
        
        all_genre_reset()
        selected_genre_reset()
        feedback_reset()
   
def display_search_music(track):

    st.write(""" Kindly listen to the provided music selection and 
             subsequently press the button to generate a list of similar tunes.
             Following this, we invite you to share your feedback. 
             If you find the music aligns well with the original piece, 
             please indicate your approval by checking the 'like' checkbox.""")
    # Display headers
    cols = st.columns([1, 1, 2, 2, 3])
    headers = ['track_id', 'genre', 'artist', 'track_title', 'play']
    for col, header in zip(cols, headers):
        col.write(header)
    
    # Display each track row with its feedback checkbox

    cols = st.columns([1, 1, 2, 2, 3])
    for col, field in zip(cols[:-1], [track['track_id'], track['track_genre_top'], track['artist_name'], track['track_title']]):
        col.write(field)
        
    file_name = f"{FILE_PATH}/{int(track['track_id']):06d}.mp3"
    cols[4].audio(file_name, format='audio/mp3')
        
                
def main():
    st.title('Welcome to AudioGenius !!!')
    st.image('webapp/images/logo9.png')

    col1, col2 = st.columns(2)

    with col1:
        selected_model = st.selectbox('Choose model:', [SIMPLE_COSINE_SIMILARITY, LIGHTGBM, CNN, LSTM], key='model')

    models = {
        LIGHTGBM: (supervised_learning_model, True),
        CNN: (cnn_model, True),
        LSTM: (lstm_model, True)
    }

    model, weighted = models.get(selected_model, (None, False))
    
    # Show sample music button
    with col2:
        button_label = "Change sample music!" if 'selected_track' in st.session_state else "Show sample music!"

    if st.button(button_label):
        sample = pd.read_csv(VALID_META_FILE).drop_duplicates(subset='track_id').sample(n=1)
        track = sample.iloc[0]
        # Store the selected track information in session state
        st.session_state.selected_track = track
        st.session_state.genre_predicted = False  # Reset this state to allow new genre prediction

    # If a track is selected, display it
    if 'selected_track' in st.session_state:
        st.success(f"Selected music: {st.session_state.selected_track['track_title']}")
        print(f"Selected music: {st.session_state.selected_track['track_title']}")
        display_search_music(st.session_state.selected_track)  # Your function to display the track

    # Predict genre button   
    
    if 'selected_track' in st.session_state and not st.session_state.genre_predicted:
        if selected_model != SIMPLE_COSINE_SIMILARITY:
            if st.button("Predict genre") and not st.session_state.genre_predicted:
                features = extract_features(st.session_state.selected_track['track_id'])
                print('features')
                print(features)
                st.session_state.X = features.drop('track_id', axis=1)
                pred = model.predict(st.session_state.X)
                print('---------------------------------------', pred)
                predicted_genre = pred[0]
                st.session_state.feature_importance = model.feature_importances_
                st.session_state.genre_predicted = True  # Update the state to show that genre has been predicted
                st.session_state.predicted_genre = predicted_genre  # Store the predicted genre
        else:
            # If we choose simple cosine-similarity as a recommendation model, we don't need to trained-model
            features = extract_features(st.session_state.selected_track['track_id'])
            st.session_state.X = features.drop('track_id', axis=1)
            
            
    # Ensure genre information is displayed if it has been predicted
    if 'genre_predicted' in st.session_state and st.session_state.genre_predicted:
        st.info(f"Predicted genre: {st.session_state.predicted_genre}")
    
    # Display the recommendation buttons only if the genre has been predicted or selected_model is SIMPLE_COSINE_SIMILARITY
    if ('genre_predicted' in st.session_state and st.session_state.genre_predicted) or ('selected_track' in st.session_state and selected_model == SIMPLE_COSINE_SIMILARITY):
        cols_buttons = st.columns(1)
        cols_buttons[0].button("Recommend similar pieces", on_click=selected_genre_submitted)
        genre = st.session_state.predicted_genre if 'predicted_genre' in st.session_state else None
        feature_importance = st.session_state.feature_importance if 'feature_importance' in st.session_state else None
        if 'selected_genre_submitted' in st.session_state and st.session_state.selected_genre_submitted:
            results = get_similar_music(
                st.session_state.selected_track['track_id'],
                st.session_state.X,
                genre=genre,
                weighted=weighted,
                feature_importance=feature_importance
            )
            show_result(st.session_state.selected_track['track_id'], selected_model, results)

            
if __name__ == "__main__":
    main()