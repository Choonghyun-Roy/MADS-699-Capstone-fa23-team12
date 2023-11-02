import streamlit as st
import os
import base64
import pandas as pd
import numpy as np
from feature_extraction import extract_features
from similarity_calculation import get_similar_music
from datetime import datetime
from pycaret.classification import * 


# Load trained model
model = load_model('xgboost_model_20231021')
N_OF_RECOMMEND = 10
    
# directory for user uploaded files
UPLOAD_HOME = 'user_uploaded_music'    
       
def show_result(st, result):

    with st.form(key='toggle_form'):
        # Loop through the DataFrame and add rows to the table HTML string
        col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,1,2,2,3,2])
        with col1:
            st.write('track_id')
        with col2:
            st.write('score')
        with col3:
            st.write('genre')
        with col4:
            st.write('artist')
        with col5:
            st.write('track_title')
        with col6:
            st.write('play')
        with col7:
            st.write('feedback')
            
        toggle_states = {} 
    
        for index, row in result.iterrows():
            col1, col2, col3, col4, col5, col6, col7 = st.columns([1,1,1,2,2,3,2])
            file_name = f"music_list/{row['track_id']:06d}.mp3"
            
            with col1:
                st.write(row['track_id'])
            with col2:
                st.write(round(row['similarity_score'],4))
            with col3:
                st.write(row['depth_1_genre_name'])
            with col4:
                st.write(row['artist_name'])
            with col5:
                st.write(row['track_title'])
            with col6:
                st.audio(file_name, format='audio/mp3')
            with col7:
                toggle_states[index] = st.checkbox('Like', key=index)
                
        submit_button = st.form_submit_button(label="Send feedback")
        
        if submit_button:
            print('do actions')


def get_base64(bin_file):
    with open(bin_file, 'rb') as f:
        data = f.read()
    return base64.b64encode(data).decode()
      
def set_background(png_file):
    bin_str = get_base64(png_file)
    page_bg_img = '''
    <style>
    .stApp {
    background-image: url("data:image/png;base64,%s");
    background-size: cover;  
    background-position: top center;  /* Set the position of the background image to the top center */
    background-repeat: no-repeat;
    }
    </style>
    ''' % bin_str
    st.markdown(page_bg_img, unsafe_allow_html=True)
            
def main():
    
    st.title('Welcome to AudioGenious !!!')
    st.image('images/logo_2.png')  
    
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
           
        # get recommendation list   
        button_clicked = st.button("Provide similar music in same genre...")
        if button_clicked:
            result = get_similar_music(file_name, X, genre_pred, N_OF_RECOMMEND)
            show_result(st, result)
            
        button_clicked_all = st.button("Provide similar music in all genre...")
        if button_clicked_all:
            result = get_similar_music(file_name, X, None, N_OF_RECOMMEND)
            show_result(st, result)
            
       
if __name__ == "__main__":
    main()