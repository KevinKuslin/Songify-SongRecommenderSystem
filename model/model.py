class SpotifyRecommender:
    def __init__(self, df, pipeline):
        self.df = df
        self.pipeline = pipeline

    def recommend(self, track_name, top_k=12):
        track_name = track_name.strip().lower()

        df = self.df.copy()
        df["track_name_clean"] = df["track_name"].str.lower().str.strip()

        matches = df[df["track_name_clean"] == track_name]
        if matches.empty:
            raise ValueError(f"Song '{track_name}' not found")

        query_song = matches.iloc[0]

        idx = query_song.name
        query = df.loc[[idx]]

        query_transformed = self.pipeline.named_steps["preprocess"].transform(query)

        deduplicate_consequence = 50

        distances, indices = self.pipeline.named_steps["knn"].kneighbors(
            query_transformed,
            n_neighbors=top_k + 1 + deduplicate_consequence
        )

        recs = df.iloc[indices[0][1:]][[
            "track_name", "artist_name", "duration_ms",
            "popularity", "acousticness", "danceability",
            "instrumentalness", "speechiness"
        ]].copy()

        recs["similarity_score"] = 1 - distances[0][1:] 

        recs = recs[
            ~(
                (recs["track_name"] == query_song["track_name"]) & 
                (recs["artist_name"] == query_song["artist_name"])
            )
        ]
        
        recs = (
            recs
            .sort_values("similarity_score", ascending=False)
            .drop_duplicates(subset=["track_name", "artist_name"])
            .head(top_k)
            .reset_index(drop=True)
        )

        return query_song.to_dict(), recs

