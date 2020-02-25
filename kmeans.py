from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.compose import ColumnTransformer
from sklearn.cluster import KMeans
import pandas as pd
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
    playlist_df['cluster'] = preds
    return playlist_df


def cluster_averages():
    clust = playlist_df.groupby('cluster').agg('mean')
    return clust


def main():
    preprocessor = pre_process()
    playlist_df_out = train_model(preprocessor)
    clusters = cluster_averages()
    return playlist_df_out, clusters


if __name__ == "__main__":
    sp = utils.set_sp_credentials()
    ids = utils.get_user_playlists(sp).id
    playlist_df = pd.DataFrame()
    for i in ids:
        temp = utils.get_playlist_tracks(sp, i)
        playlist_df = playlist_df.append(temp)
    playlist_df = playlist_df.reset_index(drop=True)
    df_train = playlist_df.drop(columns=['music', 'artist', 'artist_url', 'music_url', 'playlist_id'])
    main()
