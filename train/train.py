import pandas as pd
import numpy as np
import joblib
from sklearn.cluster import MiniBatchKMeans
from preprocess import run_preprocessing

# Load dataset
df_raw = pd.read_csv("dataset/SpotifyFeatures.csv")

# Jalankan preprocessing sesuai logika awal
df, X_scaled, audio_features, scaler = run_preprocessing(df_raw) 

# Artist Embeddings
artist_embeddings = (
    df.assign(idx=df.index)
        .groupby('artist_name')
        .apply(lambda x: X_scaled[x['idx']].mean(axis=0))
)

# Ubah format vektor menjadi matrix 2D
artist_matrix = np.vstack(artist_embeddings.values)

# Ambil nama-nama artist
artist_names = artist_embeddings.index.to_list()

# Clustering
k = 26
kmeans = MiniBatchKMeans(n_clusters=k, batch_size=5000, random_state=42)
df['cluster'] = kmeans.fit_predict(X_scaled)

# Simpan semua artefak
artifacts = {
    'df': df,
    'audio_features': audio_features,
    'scaler': scaler,
    'artist_matrix': artist_matrix,
    'artist_names': artist_names,
    'kmeans': kmeans
}

joblib.dump(artifacts, "model/spotify_model.pkl") 