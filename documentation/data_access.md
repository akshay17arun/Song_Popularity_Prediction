# Data Access Documentation

## Song Information

Song data can be found at [here](../data/all_tracks_clean.csv). 

Data was last updated on March 9th, 2026.

### Features:

* **genre**: Genre of the song. A list of all genres and their corresponding Deezer IDs can be found [here](genres.md)
* **id**: Deezer's unique track ID
* **title**: Song's title
* **artist**: Song's Artist
* **album**: Album that features that song
* **duration_sec**: Duration of the song (sec)
* **rank**: Song popularity rank assigned by Deezer (higher rank value corresponds to higher popularity)
* **explicit**: Boolean variable for whether the song is explicit

## Song Snippet Files

Since Deezer's API only grants temporary links to download song snippets (the first 30 seconds). Songs must be downloaded soon after the link is generated. To download a song snippet as a .wav file:

1. Ensure the following python packages are installed:
    * requests
    * sys
    * pydub
2. Locate the Song ID from the [song dataframe](../data/all_tracks_clean.csv)
3. Run the following command in terminal from the base directory, replacing {id} with the song ID and {file_name} with whatever you want to name the file: 
```
python scripts/song_download.py {id} {file_name}.wav 
```

The song snippet will be saved to the base directory