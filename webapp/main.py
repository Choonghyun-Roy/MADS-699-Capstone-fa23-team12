import streamlit as st
import os
import base64
import pandas as pd
import numpy as np
from similarity_calculation import get_similar_music
from datetime import datetime
from pycaret.classification import * 
from feature_extraction import extract_features
from dbconnection import execute_query

# Load trained model
xgboost_model = load_model('xgboost_model_20231021')
cnn_model = load_model('xgboost_model_20231021')
N_OF_RECOMMEND = 10
    
# directory for user uploaded files
UPLOAD_HOME = 'webapp/user_uploaded_music'    
  
def insert_feedback(user_name, selected_model, selected_metric, org_track_id, track_id, like_yn):
    insert_query = """
        INSERT INTO user_feedback (user_name, model_type, metric_type, original_track_id, recommended_track_id, like_yn) 
        VALUES (%s, %s, %s, %s, %s, %s);
    """
    data = (user_name, selected_model, selected_metric, org_track_id, track_id, like_yn)
    execute_query(insert_query, data)

def all_genre_submitted():
    st.session_state.all_genre_submitted = True
    
def all_genre_reset():
    st.session_state.all_genre_submitted = False
    
def selected_genre_submitted():
    st.session_state.selected_genre_submitted = True
    
def selected_genre_reset():
    st.session_state.selected_genre_submitted = False

def submitted():
    st.session_state.submitted = True
    
def reset():
    st.session_state.submitted = False
                      
def show_result(exist_yn, org_track_id, selected_model, selected_metric, result):
    if exist_yn:
        st.info('The music already exists in our database!')
    else:
        st.info('The music is new. Will be added to our database!')
          
    # Use session state to store toggle states
    if 'toggle_states' not in st.session_state:
        st.session_state.toggle_states = {}
        
    with st.form(key='feedback_form'):
        # Display headers
        cols = st.columns([1, 1, 1, 2, 2, 3, 2])
        headers = ['track_id', 'score', 'genre', 'artist', 'track_title', 'play', 'feedback']
        for col, header in zip(cols, headers):
            col.write(header)
        
        # Display each track row with its feedback checkbox
        for index, row in result.iterrows():
            cols = st.columns([1, 1, 1, 2, 2, 3, 2])
            track_id = row['track_id']
            for col, field in zip(cols[:-2], [track_id, round(row['similarity_score'], 4), row['depth_1_genre_name'], row['artist_name'], row['track_title']]):
                col.write(field)
            file_name = f"webapp/music_list/{int(track_id):06d}.mp3"
            cols[5].audio(file_name, format='audio/mp3')
            feedback_key = f"like_{track_id}"
            liked = cols[6].checkbox('Like', key=feedback_key, value=st.session_state.toggle_states.get(feedback_key, False))
            st.session_state.toggle_states[feedback_key] = liked  # Update session state
        
        user_name = st.text_input("Enter your username", key="user_name")
        st.form_submit_button("Send feedback", on_click=submitted)

    if 'submitted' in st.session_state and st.session_state.submitted:
        print('submitted')
        with st.spinner("Processing feedback..."):
            # Process feedback for each track using the session state information
            for track_id, liked in st.session_state.toggle_states.items():
                numeric_track_id = int(track_id.split('_')[1])
                insert_feedback(user_name, selected_model, selected_metric, str(int(org_track_id)).zfill(6), str(int(numeric_track_id)).zfill(6), liked)
            st.success('Thank you!!! Your feedback is successfully submitted!')
        
        # Clear feedback after submission
        for key in st.session_state.toggle_states.keys():
            st.session_state.toggle_states[key] = False    
        
        all_genre_reset()
        selected_genre_reset()
        reset()
            
def main():
    
    st.title('Welcome to AudioGenious !!!')
    st.image('webapp/images/logo5.png')  
    
    col1, col2 = st.columns(2)

    # Place a select box in the first column
    with col1:
        selected_model = st.selectbox('Choose model: ', ['Xgboost', 'CNN'], key='model')

    # Place another select box in the second column
    with col2:
        selected_metric = st.selectbox("Choose metric:", ['Cosine-similarity','Weighted cosine-similarity'], key='metric')

    # Output the selected options
    st.write(f"Model: {selected_model}")
    st.write(f"Metric: {selected_metric}")

    model = xgboost_model
    if selected_model == 'CNN':
       model = cnn_model
    weighted = False
    if selected_metric == 'Weighted cosine-similarity':
       weighted = True
    
    # upload music_file
    uploaded_file = st.file_uploader("Choose your favorite music file!")
    
    if uploaded_file is not None:
        # show play tool
        st.audio(uploaded_file, format='audio/mp3')
        
        # save the file
        now = datetime.now().strftime('%Y%m%d%H%M%S')
        upload_location = f"{UPLOAD_HOME}/{now}"
        
        if not os.path.exists(upload_location):
            os.makedirs(upload_location)
            
        file_name = uploaded_file.name
        file_path = os.path.join(upload_location, file_name)
        
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        st.success(f"Music uploaded successfully. Now analyzing the music...")
        
        # extract features from the file
        features = extract_features(file_path)
        st.success("Successfully extracted features !!!")
        
        # predict genre
        X = features.drop('track_id', axis=1)
        genre_pred = model.predict(X)[0]
        st.info(f"Predicted genre : {genre_pred}")
      
        # get the feature_importances to calculate weighted cosine-similarity
        feature_importance = model.feature_importances_
        
        st.button("Provide similar music in same genre...", on_click=selected_genre_submitted)
        st.button("Provide similar music in all genre...", on_click=all_genre_submitted)
          
        # get recommendation list   
        if 'selected_genre_submitted' in st.session_state and st.session_state.selected_genre_submitted:
            exist_yn, track_id, result = get_similar_music(upload_location, file_name, X, genre_pred, N_OF_RECOMMEND, weighted, feature_importance)
            show_result(exist_yn, track_id, selected_model, selected_metric, result)
            
        if 'all_genre_submitted' in st.session_state and st.session_state.all_genre_submitted:
            exist_yn, track_id, result = get_similar_music(upload_location, file_name, X, None, N_OF_RECOMMEND, weighted, feature_importance)
            show_result(exist_yn, track_id, selected_model, selected_metric, result)
            
       
if __name__ == "__main__":
    main()