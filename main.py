import pandas as pd
import utils
import config


def main():
    # get playlist id
    ids = utils.get_user_playlists(sp, login).id

    # get dataframe with all tracks from playlist
    playlist_tracks_df = pd.DataFrame()

    for i in ids:
        temp = utils.get_playlist_tracks(sp, i)
        playlist_tracks_df = pd.concat([playlist_tracks_df, temp])

    playlist_tracks_df = playlist_tracks_df.reset_index(drop=True)

    # k-means clustering flow
    preprocessor = utils.pre_process()
    playlist_df_clusters = utils.train_model(preprocessor, playlist_tracks_df, 7)
    clusters = utils.cluster_averages(playlist_df_clusters).sort_values('danceability')
    clusters['duration_seconds'] = clusters.duration_ms / 1000 / 60

    print(clusters)

    return


if __name__ == "__main__":
    # establish connection to spotify data via spotify api
    login = config.user
    sp = utils.set_sp_credentials()

    main()
