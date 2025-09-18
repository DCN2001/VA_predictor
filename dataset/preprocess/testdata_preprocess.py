import os 
import numpy as np
import librosa
import argparse
import torch
import soundfile as sf
from BEATs.BEATs import BEATs, BEATsConfig
from tqdm import tqdm

def testdata_preprocess(args):
    #Datapath
    org_path = args.testpath
    sample_path = args.sample_path
    feature_path = args.feature_path

    #BEATs model setup
    checkpoint = torch.load("./BEATs/BEATs.pt",weights_only=True)
    cfg = BEATsConfig(checkpoint['cfg'])
    BEATs_model = BEATs(cfg).to('cuda') 
    BEATs_model.predictor = None
    BEATs_model.load_state_dict(checkpoint['model'])
    BEATs_model.eval() 

    #Start processing
    song_list = os.listdir(org_path)
    for song in tqdm(song_list):
        song_path = os.path.join(org_path,song)
        full_song, sr = librosa.load(song_path, sr=16000)
        #Cut the segment from the middle 30 second
        full_len = len(full_song) 
        sample_len = 30 * sr
        half = int(full_len / 2)
        start = half - int(sample_len/2)
        end = start + sample_len
        sample_song = full_song[start:end] 
        sf.write(os.path.join(sample_path,song), sample_song, sr)
        #Get the feature of those segments
        segment_samples = 5 * sr        #5 seconds a segment
        num_segments = len(sample_song) // segment_samples
        song_features = []
        for i in range(num_segments):
            start = i * segment_samples
            end = (i + 1) * segment_samples
            segment = sample_song[start:end]
            segment_tensor = torch.tensor(segment, dtype=torch.float32).unsqueeze(0).to('cuda')     #Transform to tensor
            # Use BEATs to extract feature of each and save as .npy
            with torch.no_grad():
                features, _ = BEATs_model.extract_features(segment_tensor)
                feature_chunk = features.mean(dim=1).squeeze(0).cpu().numpy()
            song_features.append(feature_chunk)
        #Save feature as .npy
        song_features = np.stack(song_features)
        np.save(os.path.join(feature_path, f"{song.replace('.mp3','')}.npy"), song_features)

#--------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--testpath', type=str, default='/mnt/gestalt/home/dcn2001/xrspace_testdata/org', help="Path of testdata folder")
    parser.add_argument('--sample_path', type=str, default='/mnt/gestalt/home/dcn2001/xrspace_testdata/sample', help="Path of 30s sample")
    parser.add_argument('--feature_path', type=str, default='/mnt/gestalt/home/dcn2001/xrspace_testdata/feature', help="Destination path for saving the feature")
    args = parser.parse_args()
    
    testdata_preprocess(args)
    