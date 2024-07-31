from flask import Flask, render_template, request
import os
import spotipy
import pandas as pd
from spotipy.oauth2 import SpotifyClientCredentials
from dotenv import load_dotenv
import matplotlib.pyplot as plt
import seaborn as sns
import io
import base64
import logging

app = Flask(__name__)

# Configure logging
logging.basicConfig(level=logging.INFO)

# Load environment variables from .env file
load_dotenv()

# Get client ID and client secret from environment variables
client_id = os.getenv("CLIENT_ID")
client_secret = os.getenv("CLIENT_SECRET")

# Check if client_id and client_secret are set correctly
if not client_id or not client_secret:
    logging.error("CLIENT_ID or CLIENT_SECRET environment variables not set")
    raise ValueError("CLIENT_ID or CLIENT_SECRET environment variables not set")

# Initialize Spotify client
spotify = spotipy.Spotify(client_credentials_manager=SpotifyClientCredentials(client_id=client_id, client_secret=client_secret))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/results', methods=['POST'])
def results():
    artist_uri = request.form['artist_uri']
    try:
        results = spotify.artist_top_tracks(artist_uri)
    except spotipy.SpotifyException as e:
        logging.error(f"Spotify API error: {e}")
        return "Error retrieving data from Spotify API", 500

    tracks_data = []
    for track in results['tracks']:
        try:
            track_id = track['id']
            track_features = spotify.audio_features(track_id)[0]
            track_info = {
                'name': track['name'],
                'popularity': track['popularity'],
                'duration_ms': track['duration_ms'],
                'valence': track_features['valence'],
                'energy': track_features['energy']
            }
            tracks_data.append(track_info)
        except Exception as e:
            logging.error(f"Error processing track {track['name']}: {e}")

    if not tracks_data:
        return "No tracks found for the artist.", 404

    df = pd.DataFrame(tracks_data)
    df['duration_seconds'] = df['duration_ms'] / 1000

    # Sort by increasing popularity and get top 3
    top_3_tracks = df.sort_values(by='popularity', ascending=True).head(3)

    # Plotting
    fig, ax = plt.subplots(figsize=(12, 8))
    scatter = sns.scatterplot(
        x='valence',
        y='popularity',
        data=df,
        hue='popularity',
        size='energy',
        sizes=(50, 300),
        palette='viridis',
        alpha=0.8,
        edgecolor='w',
        linewidth=0.5
    )
    max_popularity = df['popularity'].max()
    min_popularity = df['popularity'].min()
    max_pop_song = df[df['popularity'] == max_popularity].iloc[0]
    min_pop_song = df[df['popularity'] == min_popularity].iloc[0]
    scatter.text(max_pop_song['valence'], max_pop_song['popularity'] + 1, max_pop_song['name'], horizontalalignment='left', size='medium', color='black', weight='semibold')
    scatter.text(min_pop_song['valence'], min_pop_song['popularity'] + 1, min_pop_song['name'], horizontalalignment='left', size='medium', color='black', weight='semibold')
    plt.title('Song Valence vs Popularity with Enhanced Visualization')
    plt.xlabel('Valence (happiness)')
    plt.ylabel('Popularity')
    plt.legend(title='Popularity and Energy')
    plt.grid(True)
    plt.tight_layout()

    # Save plot to a bytes buffer
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    plot_url = base64.b64encode(img.getvalue()).decode()

    return render_template('results.html', top_3_tracks=top_3_tracks, plot_url=plot_url)

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
