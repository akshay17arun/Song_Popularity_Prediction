"""
Access Deezer's public API to fetch top tracks from various genres. 

Saves each genre's tracks in a separate CSV file under the "data" directory, 
and also combines all tracks into a single CSV file for easier analysis.

"""

import requests
import csv
import time
import os

os.makedirs("data", exist_ok=True)

GENRES = {
    132: "Pop",
    116: "Rap_Hip_Hop",
    122: "Reggaeton",
    152: "Rock",
    113: "Dance",
    165: "RnB",
    85:  "Alternative",
    186: "Christian",
    106: "Electro",
    466: "Folk",
    144: "Reggae",
    129: "Jazz",
    84:  "Country",
    67:  "Salsa",
    65:  "Traditional_Mexicano",
    98:  "Classical",
    173: "Films_Games",
    464: "Metal",
    169: "Soul_Funk",
    2:   "African_Music",
    16:  "Asian_Music",
    153: "Blues",
    75:  "Brazilian_Music",
    71:  "Cumbia",
    81:  "Indian_Music",
    95:  "Kids",
    197: "Latin_Music",
}

FIELDS = ["id", "title", "artist", "album", "duration_sec", "rank", "explicit"]
TARGET  = 500
LIMIT   = 100  # max per request

def fetch_tracks(genre_id, index, limit):
    url = f"https://api.deezer.com/chart/{genre_id}/tracks"
    r = requests.get(url, params={"limit": limit, "index": index})
    if r.status_code != 200:
        return []
    data = r.json()
    return data.get("data", [])

def track_to_row(t):
    return {
        "id":           t.get("id"),
        "title":        t.get("title"),
        "artist":       t["artist"]["name"],
        "album":        t["album"]["title"],
        "duration_sec": t.get("duration"),
        "rank":         t.get("rank"),
        "explicit":     t.get("explicit_lyrics")
    }

for genre_id, genre_name in GENRES.items():
    filepath = f"data/{genre_name}.csv"
    tracks = []
    seen_ids = set()
    index = 0

    print(f"[{genre_name:<22}] fetching...", end=" ", flush=True)

    while len(tracks) < TARGET:
        batch = fetch_tracks(genre_id, index, LIMIT)
        if not batch:
            break
        for t in batch:
            if t["id"] not in seen_ids:
                seen_ids.add(t["id"])
                tracks.append(track_to_row(t))
        if len(batch) < LIMIT:
            break  # no more pages
        index += LIMIT
        time.sleep(0.3)

    tracks = tracks[:TARGET]

    with open(filepath, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(tracks)

    print(f"{len(tracks):>3} tracks saved → {filepath}")

print("\nCombining all CSVs...")

combined_path = "data/all_tracks.csv"
with open(combined_path, "w", newline="", encoding="utf-8") as f:
    writer = csv.DictWriter(f, fieldnames=["genre"] + FIELDS)
    writer.writeheader()
    for genre_id, genre_name in GENRES.items():
        filepath = f"data/{genre_name}.csv"
        with open(filepath, newline="", encoding="utf-8") as gf:
            reader = csv.DictReader(gf)
            for row in reader:
                writer.writerow({"genre": genre_name, **row})

print(f"Combined CSV saved → {combined_path}")
print("Done!")