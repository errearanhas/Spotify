from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import utils


sp = utils.set_sp_credentials()
playlist_df = utils.get_playlist_tracks(sp)

scale_minmax = Pipeline(steps=[
    ('scaling', MinMaxScaler())
    ])

# Compondo os pré-processadores
preprocessor = ColumnTransformer(transformers=[
    ('scale', scale_minmax, ['loudness', 'tempo', 'duration_ms'])
    ])

model = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('kmeans', KMeans(n_clusters=7, random_state=42))
])


df_train = playlist_df.drop(columns=['music', 'artist', 'artist_url', 'music_url', 'playlist_id'])
Kmeans = model.fit(df_train)

# Training and Predicting
preds = Kmeans.predict(df_train)

# Adding predictions to dataframe
playlist_df['cluster'] = preds

# Grouping clusters to see the averages
clusters = playlist_df.groupby('cluster').agg('mean')
