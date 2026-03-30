import pandas as pd
import librosa
import requests
from pydub import AudioSegment
import numpy as np
import os
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm

CNN_HEIGHT = 128
CNN_WIDTH  = 128
CHANNELS   = 1

df = pd.read_csv("../data/all_tracks_clean.csv")

def spectrogram(row_num, show=False, verbose=False):
    """
    Generate a mel spectrogram of a track given the row number in the dataframe

    If show is True, also display the spectrogram using matplotlib

    Outputs a tensor of 
    """

    if verbose:
        print(f"Processing row {row_num} - {df.loc[row_num]['artist']} - {df.loc[row_num]['title']}")

    # Get Track URL from row_num
    row = df.loc[row_num]
    req = requests.get(f"https://api.deezer.com/track/{row['id']}").json()
    preview_url = req['preview']
    if not preview_url:
        return None
    
    # Get .wav file using Deezer preview URL and pydub

    tmp_mp3 = f"tmp_{row['id']}.mp3"
    tmp_wav = f"tmp_{row['id']}.wav"
    r = requests.get(preview_url)
    with open(tmp_mp3, "wb") as f:
        f.write(r.content)
    
    AudioSegment.from_mp3(tmp_mp3).export(tmp_wav, format="wav")
    os.remove(tmp_mp3)


    y, sr = librosa.load(tmp_wav, sr=22050, duration=30, mono=True)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=CNN_HEIGHT, n_fft=2048, hop_length=512, fmax=8000)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_norm_uint8 = cv2.normalize(mel_db, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    mel_resized    = cv2.resize(mel_norm_uint8, (CNN_WIDTH, CNN_HEIGHT),
                                interpolation=cv2.INTER_LINEAR)
    
    mel_final = mel_resized.astype(np.float32) / 255.0
    mel_final = mel_final[np.newaxis, ...] # Pytorch puts channel dimension at the front

    os.remove(tmp_wav)

    if (show):
        plt.figure(figsize=(10, 4))
        librosa.display.specshow(mel_db, sr=sr, x_axis="time", y_axis="mel", fmax=8000)
        plt.colorbar(format="%+2.0f dB")
        plt.title(f"{row['artist']} - {row['title']}\nRank: {row['rank']}")
        plt.tight_layout()
        plt.show()

    return mel_final



spectrograms = []
valid_indices = []


# Stack images into a tensor, skipping over any tracks that failed to generate a spectrogram
for i in tqdm(df.index, desc="Generating spectrograms"):
    result = spectrogram(i)
    if result is not None:
        spectrograms.append(result)
        valid_indices.append(i)

X = np.stack(spectrograms)
valid_indices = np.array(valid_indices)

np.savez("../data/spectrogram_tensors.npz", X=X, indices=valid_indices)

print(f"Saved spectrogram tensors to ../data/spectrogram_tensors.npz with shape {X.shape}")