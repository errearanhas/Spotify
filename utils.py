import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
import config


cli_id = config.client_id
secret = config.client_secret
my_playlist = config.playlist_id
usr = config.user


def set_sp_credentials(client_id=cli_id,
                       client_secret=secret):
    # API Login
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    sp = spotipy.Spotify(client_credentials_manager=client_credentials_manager)  # spotify object to access API
    return sp


def get_playlist_tracks(sp, playlist_id=my_playlist):
    tracks = sp.user_playlist_tracks(playlist_id=playlist_id)
    tracks_uri_list = [x['track']['uri'] for x in tracks['items']]

    # Extracting song and artist names
    songs = [i['track']['name'] for i in tracks['items']]
    artist = [i['track']['artists'][0]['name'] for i in tracks['items']]
    artist_url = [i['track']['artists'][0]['external_urls']['spotify'] for i in tracks['items']]
    music_url = [i['track']['external_urls']['spotify'] for i in tracks['items']]
    data = {'music': songs, 'artist': artist, 'artist_url': artist_url,
            'music_url': music_url, 'playlist_id': playlist_id}

    # Getting track features
    features = []
    for i in tracks_uri_list:
        features += sp.audio_features(i)

    # Creating playlist dataframe
    cols_to_drop = ['id', 'analysis_url', 'key', 'time_signature', 'track_href', 'type', 'uri', 'mode']

    feat = pd.DataFrame(features).drop(cols_to_drop, axis=1)
    meta = pd.DataFrame(data)

    playlist_df = pd.concat([feat, meta], axis=1)
    return playlist_df


def get_user_playlists(sp, login=usr):
    user = sp.user_playlists(login)

    total = user['total']

    names = [user['items'][i]['name'] for i in range(total)]
    ids = [user['items'][i]['id'] for i in range(total)]
    url = [user['items'][i]['external_urls']['spotify'] for i in range(total)]
    # desc = [user['items'][i]['description'] for i in range(total)]
    colab = [user['items'][i]['collaborative'] for i in range(total)]

    d = {'id': ids, 'names': names, 'url': url, 'colab': colab}
    user_playlist_df = pd.DataFrame(data=d)

    return user_playlist_df
