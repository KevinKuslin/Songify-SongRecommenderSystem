import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler

def winsorize(df, col):
    q1 = df[col].quantile(0.25)
    q3 = df[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    df[col] = df[col].clip(lower, upper)
    return df

def run_preprocessing(df):
    # 1. Winsorizing
    df = winsorize(df, 'duration_ms')
    df = winsorize(df, 'instrumentalness')

    # 2. Key Mapping & Sin/Cos
    key_map = {
        'C': 0, 'C#': 1, 'Db': 1, 'D': 2, 'D#': 3, 'Eb': 3, 'E': 4,
        'F': 5, 'F#': 6, 'Gb': 6, 'G': 7, 'G#': 8, 'Ab': 8, 'A': 9,
        'A#': 10, 'Bb': 10, 'B': 11
    }
    df['key'] = df['key'].map(key_map)
    df['key_sin'] = np.sin(2 * np.pi * df['key'] / 12)
    df['key_cos'] = np.cos(2 * np.pi * df['key'] / 12)
    df.drop(columns=['key'], inplace=True)

    # 3. Mode Mapping
    df['mode'] = df['mode'].map({'Major': 1, 'Minor': 0})

    # 4. One Hot Encoder Genre
    ohe = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    ohe_genre = ohe.fit_transform(df[['genre']])
    ohe_df = pd.DataFrame(ohe_genre, columns=ohe.get_feature_names_out(['genre']), index=df.index)
    df = pd.concat([df, ohe_df], axis=1)

    # 5. Feature Selection & Scaling
    audio_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df[audio_features])

    return df, X_scaled, audio_features, scaler 