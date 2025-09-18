import os
import librosa
import argparse
import pandas as pd
import numpy as np
import torch
from BEATs.BEATs import BEATs, BEATsConfig
from tqdm import tqdm

#VA label denormalize function
def label_denormalize(x):
    return x * 8 + 1

def load_id_list(file_path):
    with open(file_path, "r") as f:
        return [f"{int(line.strip())}.mp3" for line in f if line.strip()]

def PMemo_preprocess(args):
    #Setting
    datapath = os.path.join(args.datapath,"PMEmo2019/chorus")
    label_df = pd.read_csv(os.path.join(args.datapath,"PMEmo2019/annotations/static_annotations.csv"))
    destpath = args.destpath
    #Create saving path for both feature and label
    os.makedirs(os.path.join(destpath,"feature","train"),exist_ok=True)
    os.makedirs(os.path.join(destpath,"feature","valid"),exist_ok=True)
    os.makedirs(os.path.join(destpath,"feature","test"),exist_ok=True)
    os.makedirs(os.path.join(destpath,"label","train"),exist_ok=True)
    os.makedirs(os.path.join(destpath,"label","valid"),exist_ok=True)
    os.makedirs(os.path.join(destpath,"label","test"),exist_ok=True)

    #Load pretrained BEATs
    checkpoint = torch.load("./BEATs/BEATs.pt",weights_only=True)
    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg).to('cuda') 
    BEATs_model.predictor = None
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval() 

    #Split dataset
    train_path = "./PMemo_split/train.txt"
    valid_path = "./PMemo_split/valid.txt"
    test_path = "./PMemo_split/test.txt"
    train_files = load_id_list(train_path)
    valid_files = load_id_list(valid_path)
    test_files  = load_id_list(test_path)

    #Preprocessed
    #Train set
    for file in tqdm(train_files):
        #Get label
        file_id = int(file.replace(".mp3",""))    
        row = label_df[label_df['musicId'] == file_id].iloc[0]
        valence_mean = row['Valence(mean)']
        arousal_mean = row['Arousal(mean)']
        #Denormalize and save the label as npy
        label_array = np.array([label_denormalize(valence_mean), label_denormalize(arousal_mean)], dtype=np.float32)
        np.save(os.path.join(destpath,"label","train", f"{file_id}.npy"), label_array)
        #Read audio
        song_path = os.path.join(datapath,file)
        full_song, sr = librosa.load(song_path, sr=16000)
        #Cut into segments
        total_samples = len(full_song)
        segment_samples = 5 * sr       
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

    #Valid set
    for file in tqdm(valid_files):
        #Get label
        file_id = int(file.replace(".mp3",""))    
        row = label_df[label_df['musicId'] == file_id].iloc[0]
        valence_mean = row['Valence(mean)']
        arousal_mean = row['Arousal(mean)']
        #Denormalize and save the label as npy
        label_array = np.array([label_denormalize(valence_mean), label_denormalize(arousal_mean)], dtype=np.float32)
        np.save(os.path.join(destpath,"label","valid", f"{file_id}.npy"), label_array)
        #Read audio
        song_path = os.path.join(datapath,file)
        full_song, sr = librosa.load(song_path, sr=16000)
        #Cut into segments
        total_samples = len(full_song)
        segment_samples = 5 * sr       
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

    #Test set
    for file in tqdm(test_files):
        #Get label
        file_id = int(file.replace(".mp3",""))    
        row = label_df[label_df['musicId'] == file_id].iloc[0]
        valence_mean = row['Valence(mean)']
        arousal_mean = row['Arousal(mean)']
        #Denormalize and save the label as npy
        label_array = np.array([label_denormalize(valence_mean), label_denormalize(arousal_mean)], dtype=np.float32)
        np.save(os.path.join(destpath,"label","test", f"{file_id}.npy"), label_array)
        #Read audio
        song_path = os.path.join(datapath,file)
        full_song, sr = librosa.load(song_path, sr=16000)
        #Cut into segments
        total_samples = len(full_song)
        segment_samples = 5 * sr       
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
        np.save(os.path.join(destpath,"feature","test", f"{file_id}.npy"), song_features)

#--------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--datapath', type=str, default='/mnt/gestalt/home/dcn2001/PMEmo', help="Path of PMemo dataset")
    parser.add_argument('--destpath', type=str, default='/mnt/gestalt/home/dcn2001/PMemo_preprocessed', help="Destination path for saving the feature")
    args = parser.parse_args()
    
    PMemo_preprocess(args)