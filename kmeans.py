from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import pandas as pd
import sys
import utils


def pre_process():
    scale_minmax = Pipeline(steps=[
        ('scaling', MinMaxScaler())
        ])

    preproc = ColumnTransformer(transformers=[
        ('scale', scale_minmax, ['loudness', 'tempo', 'duration_ms'])
        ])
    return preproc


def train_model(preproc):
    model = Pipeline(steps=[
        ('preprocessor', preproc),
        ('kmeans', KMeans(n_clusters=7, random_state=42))
    ])

    kmeans = model.fit(df_train)
    preds = kmeans.predict(df_train)
    playlist_tracks_df['cluster'] = preds
    return playlist_tracks_df


def cluster_averages():
    clust = playlist_tracks_df.groupby('cluster').agg('mean')
    return clust


def main():
    preprocessor = pre_process()
    playlist_df_out = train_model(preprocessor)
    clusters = cluster_averages().sort_values('danceability')
    clusters['duration_ms'] = clusters.duration_ms/1000/60
    clusters = clusters.rename(columns={'duration_ms': 'duration_sec'})
    # print(playlist_df_out)
    print(clusters)
    return playlist_df_out, clusters


if __name__ == "__main__":
    login = sys.argv[1]

    sp = utils.set_sp_credentials()
    ids = utils.get_user_playlists(sp, login).id

    playlist_tracks_df = pd.DataFrame()
    for i in ids:
        temp = utils.get_playlist_tracks(sp, i)
        playlist_tracks_df = playlist_tracks_df.append(temp)

    playlist_tracks_df = playlist_tracks_df.reset_index(drop=True)
    df_train = playlist_tracks_df.drop(columns=['music', 'artist', 'artist_url', 'music_url', 'playlist_id'])
    main()
