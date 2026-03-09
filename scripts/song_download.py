import sys
import requests
from pydub import AudioSegment

if len(sys.argv) != 3:
    print("Usage: python song_download.py {id} {output_path}")
    sys.exit(1)

track_id  = sys.argv[1]
out_path  = sys.argv[2]

track = requests.get(f"https://api.deezer.com/track/{track_id}").json()
preview_url = track.get("preview")

if not preview_url:
    print(f"No preview available for track {track_id}")
    sys.exit(1)

tmp_mp3 = f"tmp_{track_id}.mp3"
r = requests.get(preview_url)
with open(tmp_mp3, "wb") as f:
    f.write(r.content)

AudioSegment.from_mp3(tmp_mp3).export(out_path, format="wav")

import os
os.remove(tmp_mp3)

print(f"Saved: {track['artist']['name']} - {track['title']} to {out_path}")
