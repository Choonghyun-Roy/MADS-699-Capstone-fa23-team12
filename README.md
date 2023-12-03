# MADS-699-Capstone-fa23-team12
#### Master of Applied Data Science, University of Michigan
#### SIADS699: Capstone Project (2023, Fall)
#### Authors
  * Choonghyun Lee (roylee@umich.edu)
  * Liang Yao (yaoliang@umich.edu)
  * Shreya Uppal (shreyau@umich.edu)
  * Tianran Lin (hazellin@umich.edu)

### Why this is cool ? Easy to dive into our work??

### Music Recommendation Model
Our project aims to develop an intelligent Music Recommendation System using data science and machine learning techniques. 
In today's digital age, the availability of vast music libraries demands efficient and personalized music recommendation systems. 
By leveraging the Free Music Archive (FMA) dataset, along with the power of the "librosa" library, 
we seek to provide users with tailored music recommendations based on their preferences and listening history.
Key Questions
* Can the model accurately identify the genre of the music? 
* Does the model effectively recommend music to users based on similarity?

### Datasets
1. [fma_medium.zip](https://os.unil.cloud.switch.ch/fma/fma_medium.zip):  25,000 tracks of 30s, 16 unbalanced genres (22 GiB)

### Quick Start
1. install python 3.9.12
2. pip install -r requirments.txt


### Flatten mp3 files in dataset
Since in the original data source, mp3 files are separated in different folders. To make it easier to load files to extract features, we need to flatten the file structures. (Mac)

```
cd /path/to/parent/directory/
```

```
mkdir fma_small_flattend # flattend_file_folder_name 
```

```
find . -type f -name "*.mp3" -exec mv {} ../fma_small_flattend/ \;
```


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

### References
```
@inproceedings{fma_dataset,
  title = {{FMA}: A Dataset for Music Analysis},
  author = {Defferrard, Micha\"el and Benzi, Kirell and Vandergheynst, Pierre and Bresson, Xavier},
  booktitle = {18th International Society for Music Information Retrieval Conference (ISMIR)},
  year = {2017},
  archiveprefix = {arXiv},
  eprint = {1612.01840},
  url = {https://arxiv.org/abs/1612.01840},
}
```

```
@inproceedings{fma_challenge,
  title = {Learning to Recognize Musical Genre from Audio},
  subtitle = {Challenge Overview},
  author = {Defferrard, Micha\"el and Mohanty, Sharada P. and Carroll, Sean F. and Salath\'e, Marcel},
  booktitle = {The 2018 Web Conference Companion},
  year = {2018},
  publisher = {ACM Press},
  isbn = {9781450356404},
  doi = {10.1145/3184558.3192310},
  archiveprefix = {arXiv},
  eprint = {1803.05337},
  url = {https://arxiv.org/abs/1803.05337},
}
```

### License
* The code in this repository is released under the MIT license.
* The metadata is released under the Creative Commons Attribution 4.0 International License (CC BY 4.0).
* We do not hold the copyright on the audio and distribute it under the license chosen by the artist.
* The dataset is meant for research purposes.
