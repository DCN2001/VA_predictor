import os
import numpy as np
import librosa
import torch
import torch.utils.data
import torchaudio.transforms as T

class VA(torch.utils.data.Dataset):
    def __init__(self,data_dir,mode):
        self.feature_dir = os.path.join(data_dir,"feature_5s",mode)   
        self.mode = mode
        self.label_dir = os.path.join(data_dir,"label",mode)
        self.track_list = os.listdir(self.label_dir)
        self.dataset = []
        #Concat to [org path, vocal path, other path, label path]
        for track in self.track_list:
            self.dataset.append([os.path.join(self.feature_dir,track),os.path.join(self.label_dir,track)])

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        data = self.dataset[idx]
        #Input
        org_feature_stack = np.load(data[0])
        if self.mode == "train":
            T = org_feature_stack.shape[0]
            #Randomly select sequence length
            seg_len = np.random.randint(1, T + 1)
            #Randomly select starting point
            start_idx = np.random.randint(0, T - seg_len + 1)
            #Select segment
            org_seg = org_feature_stack[start_idx: start_idx + seg_len]
            #Mean pooling
            org_features = org_seg.mean(axis=0)  # shape: (D,)
        else:
            org_features = org_feature_stack.mean(axis=0)    #Shape: (527,)
        #Label
        label_va = np.load(data[1])
        # print(org_features.shape,label_va)
        return org_features, label_va
    
def load_data(Trainset_Option, DEAM_datapath, Emomusic_datapath, PMemo_datapath, Mood_datapath, batch_size, num_workers):
    #Build dataset
    DEAM_train_ds = VA(DEAM_datapath, "train")
    DEAM_valid_ds = VA(DEAM_datapath, "valid")
    Emomusic_train_ds = VA(Emomusic_datapath, "train")
    Emomusic_valid_ds = VA(Emomusic_datapath, "valid")
    PMemo_train_ds = VA(PMemo_datapath, "train")
    PMemo_valid_ds = VA(PMemo_datapath, "valid")
    Mood_train_ds = VA(Mood_datapath, "train")
    Mood_valid_ds = VA(Mood_datapath, "valid")

    #Set combination
    if Trainset_Option == "DEPM":
        train_ds = torch.utils.data.ConcatDataset([DEAM_train_ds, Emomusic_train_ds, PMemo_train_ds, Mood_train_ds])
        valid_ds = torch.utils.data.ConcatDataset([DEAM_valid_ds, Emomusic_valid_ds, PMemo_valid_ds, Mood_valid_ds])
    elif Trainset_Option == "DEP_only":
        train_ds = torch.utils.data.ConcatDataset([DEAM_train_ds, Emomusic_train_ds, PMemo_train_ds])
        valid_ds = torch.utils.data.ConcatDataset([DEAM_valid_ds, Emomusic_valid_ds, PMemo_valid_ds])
    elif Trainset_Option == "M_only":
        train_ds = Mood_train_ds
        valid_ds = Mood_valid_ds

    #Build dataloader
    trainset_loader = torch.utils.data.DataLoader(dataset=train_ds,
                                                  batch_size=batch_size,
                                                  pin_memory=True,
                                                  shuffle=True,
                                                  drop_last=True,
                                                  num_workers=num_workers)
    
    validset_loader = torch.utils.data.DataLoader(dataset=valid_ds,
                                                  batch_size=batch_size,
                                                  pin_memory=True,
                                                  shuffle=False,
                                                  drop_last=True,
                                                  num_workers=num_workers)
    return trainset_loader, validset_loader



# ## Dataloder testbench
# from tqdm import tqdm
# train_loader, valid_loader = load_data("/mnt/gestalt/home/dcn2001/DEAM_preprocessed","/mnt/gestalt/home/dcn2001/Mood_preprocessed","sadasd",1,16)
# for idx, batch in enumerate(tqdm(train_loader, desc="Train bar", colour="#9F35FF")):
#     pass