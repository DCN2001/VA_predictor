import os
import librosa
import argparse
import random
import pandas as pd
import numpy as np
import torch
from scipy.io import loadmat
from BEATs.BEATs import BEATs, BEATsConfig
from tqdm import tqdm

def label_denormalize(x):
    return (x + 1) * 4 + 1   # 把 [-1,1] → [1,9]

def AMG1838_preprocess(args):
    #Setting
    datapath = os.path.join(args.datapath,"Source/amg1838_mp3_22050_128k_30s")
    label_df = pd.read_csv(os.path.join(args.datapath,"Source/amg1838_v2.csv"))
    destpath = args.destpath
    #Create saving path for both feature and label
    os.makedirs(os.path.join(destpath,"feature","train"),exist_ok=True)
    os.makedirs(os.path.join(destpath,"feature","valid"),exist_ok=True)
    os.makedirs(os.path.join(destpath,"label","train"),exist_ok=True)
    os.makedirs(os.path.join(destpath,"label","valid"),exist_ok=True)

    #Load pretrained BEATs
    checkpoint = torch.load("./BEATs/BEATs.pt",weights_only=True)
    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg).to('cuda') 
    BEATs_model.predictor = None
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval() 

    #Split to train/valid as 85/15
    all_files = os.listdir(datapath)
    random.shuffle(all_files)
    split_idx = int(len(all_files) * 0.85)
    train_files = all_files[:split_idx]
    valid_files = all_files[split_idx:]

    #Preprocessed
    #Train
    for file in tqdm(train_files):
        file_id = int(file.replace(".mp3",""))
        #Get label
        row = label_df[label_df['id'] == file_id].iloc[0]
        valence = row['valence']
        arousal = row['arousal']
        label_array = np.array([label_denormalize(valence), label_denormalize(arousal)], dtype=np.float32)
        # Save label as npy
        np.save(os.path.join(destpath,"label","train", f"{file_id}.npy"), label_array)
        #Read audio
        song_path = os.path.join(datapath,file)
        full_song, sr = librosa.load(song_path, sr=16000)
        #Cut into segments
        total_samples = len(full_song)
        segment_samples = 5 * sr       #5 seconds a segment
        num_segments = total_samples // segment_samples
        song_features = []
        for i in range(num_segments):
            start = i * segment_samples
            end = (i + 1) * segment_samples
            segment = full_song[start:end]
            segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to('cuda')     #Transform to tensor
            # Use BEATs to extract feature of each and save as .npy
            with torch.no_grad():
                features, _ = BEATs_model.extract_features(segment_tensor)
                feature_chunk = features.mean(dim=1).squeeze(0).cpu().numpy()
            song_features.append(feature_chunk)
        #Save feature as .npy
        song_features = np.stack(song_features)
        np.save(os.path.join(destpath,"feature","train", f"{file_id}.npy"), song_features)

    #Valid
    for file in tqdm(valid_files):
        file_id = int(file.replace(".mp3",""))
        #Get label
        row = label_df[label_df['id'] == file_id].iloc[0]
        valence = row['valence']
        arousal = row['arousal']
        label_array = np.array([label_denormalize(valence), label_denormalize(arousal)], dtype=np.float32)
        # Save label as npy
        np.save(os.path.join(destpath,"label","valid", f"{file_id}.npy"), label_array)
        #Read audio
        song_path = os.path.join(datapath,file)
        full_song, sr = librosa.load(song_path, sr=16000)
        #Cut into segments
        total_samples = len(full_song)
        segment_samples = 5 * sr       #5 seconds a segment
        num_segments = total_samples // segment_samples
        song_features = []
        for i in range(num_segments):
            start = i * segment_samples
            end = (i + 1) * segment_samples
            segment = full_song[start:end]
            segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to('cuda')     #Transform to tensor
            # Use BEATs to extract feature of each and save as .npy
            with torch.no_grad():
                features, _ = BEATs_model.extract_features(segment_tensor)
                feature_chunk = features.mean(dim=1).squeeze(0).cpu().numpy()
            song_features.append(feature_chunk)
        #Save feature as .npy
        song_features = np.stack(song_features)
        np.save(os.path.join(destpath,"feature","valid", f"{file_id}.npy"), song_features)

#--------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='/mnt/gestalt/database/mood_AMG1838_1608', help="Path of AMG1838 dataset")
    parser.add_argument('--destpath', type=str, default='/mnt/gestalt/home/dcn2001/Mood_preprocessed', help="Destination path for saving the feature")
    args = parser.parse_args()
    
    AMG1838_preprocess(args)