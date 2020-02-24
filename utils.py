import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
import config

id = config.client_id
secret = config.client_secret
my_playlist = config.playlist_id

def set_sp_credentials(client_id=id,
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

    # Getting features of tracks
    features = []
    for i in tracks_uri_list:
        features += sp.audio_features(i)

    # Creating feature dataframe
    cols_to_drop = ['id', 'analysis_url', 'key', 'time_signature', 'track_href', 'type', 'uri', 'mode']

    feat = pd.DataFrame(features).drop(cols_to_drop, axis=1)
    meta = pd.DataFrame(data)

    playlist_df = pd.concat([feat, meta], axis=1)
    return playlist_df
