knn = NearestNeighbors(
    n_neighbors=20,
    metric='cosine'
)

pipeline = Pipeline(steps=[
    ('preprocess', preprocess),
    ('knn', knn)
])

pipeline.fit(df) 