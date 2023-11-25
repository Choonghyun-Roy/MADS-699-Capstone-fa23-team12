'''
@ Author : Roy Choonghyun Lee
@ Flow
1. Select / Change Model 
  - Default : Simple Cosine-Similarity
  - Action 
    . Set st.session_state.previous_selection to selected_model (for example. CNN, LSTM, LightGBM...)
    . selected_track_reset() # Reset Selected Track
    . provide_similar_music_reset() # Reset Simialr Music List
    . feedback_reset() # Reset User Feedback
    
2. Click "Provide Sample Music Piece"
  - Action
    . st.session_state.selected_track = track
    . st.session_state.genre_predicted = False # Reset this state to allow new genre prediction
    
3. If Model is not Cosine-Similarity and selected_track exists,
  - Click 'Predict Genre' 
  - Action
    . st.session_state.X = features.drop('track_id', axis=1)
    . st.session_state.genre_predicted = True  # Update the state to show that genre has been predicted
    . st.session_state.feature_importance = model.feature_importances_
    . st.session_state.predicted_genre = predicted_genre  # Store the predicted genre
    . provide the similar music list
    
4. If Model is Cosine-Similarity and selected_track exists,
  - Click 'Provide 10 Similar Music Pieces'
  - Action
    . st.session_state.provide_similar_music_reset = True (selected_genre_submitted)
    . st.session_state.X = features.drop('track_id', axis=1)
    . provide the similar music list
    
5. Feedback Submitted
  - Click "Send Feedback"
  - Action
    . selected_track_reset()
    . provide_similar_music_reset()
    . feedback_reset()
    
'''
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

SIMPLE_COSINE_SIMILARITY = 'Type-1. Simple Cosine Similarity'
LIGHTGBM = 'Type-2. LightGBM'
CNN = 'Type-3. CNN'
LSTM = 'Type-4. LSTM'

# Load trained model
def load_models():
    return  load_model('lightgbm_model_25K_88F_231124'), load_model('xgboost_model_25K_74F_231111'), load_model('xgboost_model_25K_74F_231111')

supervised_learning_model, cnn_model, lstm_model = load_models()

N_OF_RECOMMEND = 10
             
def insert_feedback(user_name, selected_model, org_track_id, track_id, like_yn):
    insert_query = """
        INSERT INTO user_feedback (user_name, model_type, original_track_id, recommended_track_id, like_yn, apply_yn) 
        VALUES (%s, %s, %s, %s, %s, 'N');
    """
    data = (user_name, selected_model, org_track_id, track_id, like_yn)
    execute_query(insert_query, data)

    
def provide_similar_music_submitted():
    st.session_state.provide_similar_music_submitted = True
    
def provide_similar_music_reset():
    st.session_state.provide_similar_music_submitted = False
  
def feedback_submitted():
    st.session_state.feedback_submitted = True
    
def feedback_reset():
    st.session_state.feedback_submitted = False

def selected_track_submitted():
    st.session_state.selected_track_submitted = True    
    
def selected_track_reset():
    st.session_state.selected_track_submitted = False

                      
def show_result(org_track_id, selected_model, result):          
    
    st.success(f"Please play the suggested music pieces, and if you find them to resemble the original sample music, tick the respective 'Similar' checkbox.")
    
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
            liked = cols[6].checkbox('Similar', key=feedback_key, value=st.session_state.toggle_states.get(feedback_key, False))
            st.session_state.toggle_states[feedback_key] = liked  # Update session state
        
        user_name = st.text_input("Enter your name and email address (ex. Roy Lee/roylee@umich.edu)", key=f"user_name_{org_track_id}_{selected_model}")
        st.form_submit_button("Send Feedback", on_click=feedback_submitted)

    if 'feedback_submitted' in st.session_state and st.session_state.feedback_submitted: 
        print('Feedback_Submitted')
        with st.spinner("Processing feedback. Please don't close the window. It will take 10 to 20 seconds."):
            # Process feedback for each track using the session state information
            for track_id, liked in st.session_state.toggle_states.items():
                numeric_track_id = int(track_id.split('_')[1])
                insert_feedback(user_name, selected_model, str(int(org_track_id)).zfill(6), str(int(numeric_track_id)).zfill(6), liked)
            st.success('Thank you very much!!! Your feedback is successfully submitted. We will use your feedback to improve our service.')
        
        # Clear feedback after submission
        for key in st.session_state.toggle_states.keys():
            st.session_state.toggle_states[key] = False    
        
        selected_track_reset()
        provide_similar_music_reset()
        feedback_reset()
        
   
def display_search_music(track):
    st.success(f"Kindly listen to the provided music piece: {st.session_state.selected_track['artist_name']} / {st.session_state.selected_track['track_title']}")
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

              
def on_model_change(selected_model):
        provide_similar_music_reset()
        feedback_reset()
        selected_track_reset()
        st.session_state.genre_predicted = False
          
def main():
    if 'previous_selection' not in st.session_state:
        st.session_state.previous_selection = None
        
    st.title('Welcome to AudioGenius !!!')
    st.write('Music Genre Prediction and Recommendation Engine Using Machine Learning')
    st.image('webapp/images/logo9.png')

    col1, col2 = st.columns(2)

    with col1:
        selected_model = st.selectbox('Choose Recommendation Model Type', [SIMPLE_COSINE_SIMILARITY, LIGHTGBM, CNN, LSTM], key='model')
        # Check if the selection has changed
        if selected_model != st.session_state.previous_selection:
            st.session_state.previous_selection = selected_model
            # Call the function you want to trigger here
            on_model_change(selected_model)
            
    models = {
        LIGHTGBM: (supervised_learning_model, True),
        CNN: (cnn_model, True),
        LSTM: (lstm_model, True)
    }

    model, weighted = models.get(selected_model, (None, False))
    
    # Show sample music button
    with col2:
        button_label = "Change a Sample Music Piece" if 'selected_track' in st.session_state else "Provide a Sample Music Piece"

    provide_music = st.button(button_label, on_click=selected_track_submitted)
    if provide_music and 'selected_track_submitted' in st.session_state and st.session_state.selected_track_submitted:
        valid_df = pd.read_csv(VALID_META_FILE)
        print('------------------------------------------------------')
        print(len(valid_df))
        print(valid_df.columns)
        
        sample = pd.read_csv(VALID_META_FILE).drop_duplicates(subset='track_id').sample(n=1)
        track = sample.iloc[0]
        # Store the selected track information in session state
        st.session_state.selected_track = track
        st.session_state.genre_predicted = False  # Reset this state to allow new genre prediction

    # If a track is selected, display it
    if 'selected_track' in st.session_state:
        print(f"Selected Music Piece: {st.session_state.selected_track['track_title']}")
        display_search_music(st.session_state.selected_track)  # Your function to display the track

    # Show 'Predict Genre' button   
    if 'selected_track' in st.session_state and not st.session_state.genre_predicted:
        if selected_model != SIMPLE_COSINE_SIMILARITY:
            if st.button("Predict Genre") and not st.session_state.genre_predicted:
                features = extract_features(st.session_state.selected_track['track_id'])
                st.session_state.X = features.drop('track_id', axis=1)
                pred = model.predict(st.session_state.X)
                predicted_genre = pred[0]
                st.session_state.genre_predicted = True  # Update the state to show that genre has been predicted
                st.session_state.feature_importance = model.feature_importances_
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
        st.button("Provide 10 Similar Music Pieces", on_click=provide_similar_music_submitted)
        genre = st.session_state.predicted_genre if 'predicted_genre' in st.session_state else None
        feature_importance = st.session_state.feature_importance if 'feature_importance' in st.session_state else None
        
    if 'provide_similar_music_submitted' in st.session_state and st.session_state.provide_similar_music_submitted:
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