import os
import random

def choose_song_cover(song):
    features = {
        "acoustic": song["acousticness"],
        "dance": song["danceability"],
        "instrumental": song["instrumentalness"],
        "speech": song["speechiness"]
    }

    feature_weights = {
        "acoustic": 0.7, 
        "dance": 1.0, 
        "instrumental": 2.0, 
        "speech": 2.0
    }

    feature_values = {
        key: features[key] * feature_weights[key]
        for key in features
    }

    dominant_feature = max(feature_values, key=feature_values.get)

    folder_path = os.path.join("static", "assets", dominant_feature)

    if not os.path.exists(folder_path):
        return "assets/default.jpg"

    images = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".png", ".jpeg"))
    ]

    if not images:
        return "assets/default.jpg"

    selected_image = random.choice(images) 

    return f"assets/{dominant_feature}/{selected_image}"
