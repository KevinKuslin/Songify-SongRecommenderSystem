from sklearn.metrics.pairwise import euclidean_distances 
import numpy as np 

class SpotifyRecommender:
    def __init__(self, df, artist_matrix, artist_names, audio_features, scaler):
        self.df = df
        self.artist_matrix = artist_matrix
        self.artist_names = artist_names
        self.audio_features = audio_features
        self.scaler = scaler
        self.track_id_to_index = dict(zip(df["track_id"], df.index)) 

    def audio_similarity_euclidean(self, song_idx, X_cluster):
        distances = euclidean_distances(
            [X_cluster[song_idx]],
            X_cluster
        )[0]

        return 1 / (1 + distances)

    def artist_similarity_euclidean(self, artist_name):
        artist_idx = self.artist_names.index(artist_name)
        artist_vec = self.artist_matrix[artist_idx]

        distances = euclidean_distances(
            [artist_vec],
            self.artist_matrix
        )[0]

        return 1 / (1 + distances)

    def recommend(self, track_name, top_k=12):
        track_name = track_name.lower().strip()

        print("INPUT DARI FORM :", repr(track_name))
        print("DF TYPE:", type(self.df), flush=True)

        df = self.df.copy()

        print("DF COLUMNS:", df.columns.tolist(), flush=True)
        print("CONTOH TRACK DI DF :", df["track_name"].head(5).tolist())
        print("CONTOH CLEAN DF :", df["track_name"].str.lower().str.strip().head(5).tolist())

        df["track_name_clean"] = df["track_name"].str.lower().str.strip()

        matches = df[df["track_name_clean"] == track_name]
        if matches.empty:
            raise ValueError(f"Song '{track_name}' not found")

        song = matches.iloc[0]
        global_idx = song.name

        cluster = song['cluster'] 

        cluster_songs = df[df['cluster'] == cluster].copy()
        cluster_songs = cluster_songs.reset_index(drop=True)

        cluster_songs['global_idx'] = cluster_songs['track_id'].map(
            self.track_id_to_index
        )

        X_cluster = self.scaler.transform(cluster_songs[self.audio_features])

        song_idx = cluster_songs.index[
            cluster_songs['track_id'] == song['track_id']
        ][0]

        audio_sims = self.audio_similarity_euclidean(song_idx, X_cluster)

        artist_sims = self.artist_similarity_euclidean(song['artist_name'])
        artist_sims_cluster = np.array([
            artist_sims[self.artist_names.index(a)]
            for a in cluster_songs['artist_name']
        ])

        w_audio = 0.7
        w_artist = 0.3 

        cluster_songs['score'] = (
            w_audio * audio_sims +
            w_artist * artist_sims_cluster
        )

        cluster_songs = cluster_songs[
            cluster_songs['track_id'] != song['track_id']
        ]

        max_per_artist = 2 
        ranked = (
            cluster_songs
            .sort_values('score', ascending=False)
            .groupby('artist_name')
            .head(max_per_artist)
            .head(top_k)
        )

        return song.to_dict(), ranked 