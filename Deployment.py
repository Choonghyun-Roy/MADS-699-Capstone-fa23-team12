from flask import Flask, render_template, request
import pandas as pd
from your_recommendation_module import recommend_songs

app = Flask(__name__)

user_song_interactions = pd.read_csv('user_song_interactions.csv')
model = load_recommendation_model()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/recommend', methods=['POST'])
def recommend():
    user_id = int(request.form['user_id'])
    recommended_songs = recommend_songs(user_id, model, user_song_interactions)
    return render_template('recommendations.html', songs=recommended_songs)

if __name__ == '__main__':
    app.run()


from flask import Flask, request, jsonify
import pandas as pd
from your_recommendation_module import recommend_songs

app = Flask(__name__)

user_song_interactions = pd.read_csv('user_song_interactions.csv')
model = load_recommendation_model()

@app.route('/recommend', methods=['POST'])
def recommendation():
    user_id = int(request.json['user_id'])
    recommended_songs = recommend_songs(user_id, model, user_song_interactions)
    return jsonify({'recommended_songs': recommended_songs})

if __name__ == '__main__':
    app.run()

