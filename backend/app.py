from flask import Flask, request, render_template, jsonify
import joblib 

from model.model import SpotifyRecommender 
from backend.util import choose_song_cover 

import os 

BASE_DIR = os.path.dirname(os.path.abspath(__file__)) 
ROOT_DIR = os.path.abspath(os.path.join(BASE_DIR, "..")) 

app = Flask(
    __name__, 
    template_folder=os.path.join(ROOT_DIR, "templates"), 
    static_folder=os.path.join(ROOT_DIR, "static")
    ) 

pipeline = joblib.load("model/pipeline.joblib") 
df = joblib.load("model/df.joblib")
df_train = joblib.load("model/df_train.joblib") 

recommender = SpotifyRecommender(df, pipeline) 

HOME_SONGS = [
    {
        "track_idx": 7,
        "track_name": "Let Me Let Go",
        "artist_name": "Laura Mayne",
        "cover": "assets/acoustic/acoustic_3.jpg"
    },
    {
        "track_idx": 2200, 
        "track_name": "This Feeling",
        "artist_name": "The Chainsmokers",
        "cover": "assets/dance/dance_1.jpg"
    },
    {
        "track_idx": 9091, 
        "track_name": "Side To Side",
        "artist_name": "Ariana Grande",
        "cover": "assets/dance/dance_5.jpg"
    }, 
    {
        "track_idx": 9103, 
        "track_name": "Delicate",
        "artist_name": "Taylor Swift",
        "cover": "assets/dance/dance_3.jpg"
    }, 
    {
        "track_idx": 98, 
        "track_name": "Forgotten Dreams",
        "artist_name": "Richard M. Sherman",
        "cover": "assets/instrumental/instrumental_3.jpg"
    }, 
    {
        "track_idx": 153, 
        "track_name": "Disconnect",
        "artist_name": "6LACK",
        "cover": "assets/speechiness/speechiness_2.jpg"
    }
]

@app.route("/")
def index():
    return render_template(
        "index.html", 
        home_songs_1=HOME_SONGS[:3], 
        home_songs_2=HOME_SONGS[3:]
        )

@app.route("/recommend", methods=["GET", "POST"])
def recommend():
    track_name = request.form.get("track_name") 

    if not track_name: 
        return render_template(
            "error.html", 
            error="Track name is required", 
            recommendations=[],
            selected_song=[]
        )
    
    try: 
        selected_song, rec_df = recommender.recommend(track_name, top_k=12) 
        selected_song["cover"] = choose_song_cover(selected_song)
        recommendations = rec_df.to_dict(orient="records") 

        for song in recommendations:
            song["cover"] = choose_song_cover(song) 

        return render_template(
            "recommendation.html", 
            recommendations=recommendations,
            selected_song=selected_song
        )
    
    except Exception as e:
        return render_template(
            "error.html", 
            error=str(e), 
            recommendations=[], 
            selected_song=[]
        )

if __name__ == "__main__":
    app.run(debug=True)