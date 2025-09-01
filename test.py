import os
import dotenv
from router import Router
dotenv.load_dotenv()


api_key = os.getenv("API_KEY")
base_url = os.getenv("BASE_URL")
print(api_key, base_url)
router = Router(api_key=api_key,
                model="gpt-oss-120b",
                base_url=base_url,
                provider="Cerebras")

@router.route(potential_queries=["play the song 'Shape of You' by Ed Sheeran", "play the playlist 'Top 100' on Spotify"])
def play_spotify(song: str = None, playlist: str = None, album: str = None, artist: str = None):
    """
    Plays music on spotify.
    """
    if song:
        print(f"Playing {song} on Spotify.")
    elif playlist:
        print(f"Playing {playlist} on Spotify.")
    elif album:
        print(f"Playing {album} on Spotify.")
    elif artist:
        print(f"Playing {artist} on Spotify.")

router.smart_call("play the song 'Shape of You' by Ed Sheeran")