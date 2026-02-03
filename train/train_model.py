from sklearn.neighbors import NearestNeighbors
from sklearn.pipeline import Pipeline 
import pandas as pd 
import joblib 

from features import preprocess
from preprocess import preprocess_data 

df = pd.read_csv('dataset/SpotifyFeatures.csv')
df_meta = df.copy() 
df_train = preprocess_data(df)

knn = NearestNeighbors(
    n_neighbors=20,
    metric='cosine'
)

pipeline = Pipeline(steps=[
    ('preprocess', preprocess),
    ('knn', knn)
])

pipeline.fit(df_train) 

# Simpen model yang dilatih 

joblib.dump(pipeline, "model/pipeline.joblib") 
joblib.dump(df_meta, "model/df.joblib") 
joblib.dump(df_train, "model/df_train.joblib") 