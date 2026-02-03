def preprocess_data(df):
    df_train = df.drop(['genre', 'track_name', 'track_id', 'artist_name', 
                        'key', 'time_signature'], axis=1).copy() 
    return df_train 