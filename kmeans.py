from sklearn.cluster import KMeans
import utils

sp = utils.set_sp_credentials()
playlist_df = utils.get_playlist_tracks(sp)

# Scaling
for col in ['loudness', 'tempo', 'duration_ms']:
    features_df[col] = ((features_fdf[col] - features_df[col].min()) / (features_df[col].max() - features_df[col].min()))

# Determining the cluster size
score_list = []
for i in range(2,10):
    kmeans_model = KMeans(n_clusters=i, random_state=3).fit(features_df)
    preds = kmeans_model.predict(features_df)
    score_list.append(kmeans_model.inertia_)

# Visualization of different cluster size performations
pd.DataFrame(score_list, index=range(2, 10)).plot(legend=False).set(xlabel="Cluster Size", ylabel="Inertia")

# Training and Predicting
kmeans_model = KMeans(n_clusters=7, random_state=3).fit(features_df)
preds = kmeans_model.predict(features_df)







# Concatenating multiple artist names
artist_list = []
for group in artists:
    artist_group = []
    for person in group:
        artist_group.append(person['name'])
    artist_list.append(', '.join(artist_group))

# Adding predictions to dataframe
features_df['cluster'] = preds

# Grouping clusters to see the averages
clusters = features_df.groupby('cluster').agg('mean')