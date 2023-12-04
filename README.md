# MADS-699-Capstone-fa23-team12
#### Master of Applied Data Science, University of Michigan
#### SIADS699: Capstone Project (2023, Fall)
#### Authors
  * Choonghyun Lee (roylee@umich.edu)
  * Liang Yao (yaoliang@umich.edu)
  * Shreya Uppal (shreyau@umich.edu)
  * Tianran Lin (hazellin@umich.edu)

### Music Recommendation Model
In the evolving landscape of digital music services, the ability to accurately recommend music tailored to individual tastes and predict musical genres from vast libraries has become paramount. This report presents an in-depth study on the development of a Music Genre Prediction and Recommendation Engine, a project that stands at the intersection of music information retrieval and machine learning. 

Central to our approach is the strategic use of the 'medium' subset from the Free Music Archive (FMA), a dataset celebrated for its comprehensive collection of diverse music tracks. This dataset's selection is pivotal, offering a rich array of music while being perfectly sized for the sophisticated analysis methods we employ, including Neural Network and Long Short-Term Memory (LSTM) networks. These advanced neural networks are particularly suited to our project's unique focus on acoustic features, capturing the complex patterns and characteristics inherent in the music.

A unique feature of our project is its primary dependence on acoustic features for recommending music. This strategy stems from the realization that users frequently come across music with minimal or no prior knowledge or context. Concentrating on the fundamental characteristics of the music, including aspects like tempo, harmony, and rhythm, our system is engineered to recommend tracks based purely on their audio characteristics, free from external influences like the music's title or producers. This method ensures a genuine and unbiased discovery process, tailored to the individual acoustic preferences of each listener.

Our project employs a three-pronged strategy for music retrieval. Firstly, we utilize cosine similarity for its effectiveness in identifying tracks with similar acoustic features. Secondly, we delve into supervised learning for genre prediction, enhanced by the additional features derived from unsupervised clustering. By applying clustering algorithms, we derive additional features from the music data, incorporating the resultant cluster labels into our supervised models. Finally, we explore the capabilities of Neural Network and LSTMs to capture both spectral and temporal aspects of music tracks, providing a comprehensive tool for music retrieval.

During the Minimum Viable Product (MVP) phase, our project employs a simple but efficient strategy: users are first given a randomly selected music track, followed by 10 tracks that are similar to the initial selection. Users then provide feedback on whether the suggested tracks are indeed similar to the initial one. This process allows us to store user feedback that will be used to refine and validate our recommendation algorithm in the future.

The project is poised for future expansion. The next phase will integrate user history into our recommendation algorithm, aiming to combine user listening patterns with music features to create a more personalized and context-aware recommendation system. This evolution will mark a significant advancement in our project, aligning with the broader goal of developing a user-centric music recommendation system. 
Additionally, in future phases of our project, we plan to use user feedback as a key component to optimize our models through a feedback loop, such as implementing the Rocchio algorithm. Currently, the limited volume of real human feedback constrains our ability to fully assess the effectiveness of this approach. However, this feedback is invaluable and will serve as a foundational element for expanding our research and incorporating such algorithms to enhance the system's performance and accuracy.

The significance of this study is manifold, offering a novel perspective in music recommendation that prioritizes the acoustic features of music. This approach presents practical methodologies for online music streaming services, aiming to enrich the user experience on digital platforms and pave the way for more intelligent, adaptive, and user-focused music discovery tools.

### Download link for audio files that used in this project
1. [fma_medium.zip](https://os.unil.cloud.switch.ch/fma/fma_medium.zip):  25,000 tracks of 30s, 16 unbalanced genres (22 GiB)

### Flatten mp3 files
Since in the original data source, mp3 files are separated in different folders. To make it easier to load files to extract features, we need to flatten the file structures. (Mac)

```
cd /path/to/parent/directory/
```

```
mkdir fma_medium_flattend # flattend_file_folder_name 
```

```
find . -type f -name "*.mp3" -exec mv {} ../fma_medium_flattend/ \;
```

### Directory Structure
```
|- data/
    |- raw                                      -> raw datasets
    |- proccessed                               -> processed datasets that used in training models
    |- resources                                -> processed datasets that used in streamlit webapp
    01. create_meta_with_genre_tree.ipynb       -> preprocess for meta data for music tracks
    02. feature_extraction.ipynb                -> extract acoustic features from the audio files
    03. create_final_datasets.ipynb             -> create final dataset for training models
|- models/
    |- final_models                             -> store final model files
    |- notebooks                                -> notebook files for training models.
|- webapp                                       -> streamlit web application
```

### Quick Start
1. install python 3.9.12
2. pip install -r requirments.txt

### Git Settings
1. Create a GitHub account if you don't already have one.
2. Go to "Settings" in your GitHub account, then select "Developer Settings," and click on "Personal access tokens."
3. Create a new personal access token.
4. Once you've created the access token, you can use it as a password when you perform Git commands, such as git push.
5. Optionally, you can save the access key by executing the following commands. This will allow you to avoid entering authentication details every time you use a git command:
```
 git config --unset credential.helper ## reset the previous setting
 git config credential.helper store ## save current account information into ~/.git_credentials
```

### How to run streamlit application
```
cd webapp
streamlit run main.py
```

### License
* The code in this repository is released under the MIT license.
* The metadata is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
* We do not hold the copyright on the audio and distribute it under the license chosen by the artist.
* The dataset is meant for research purposes.
