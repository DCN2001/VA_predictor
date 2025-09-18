from model.linear_va import FF_model           #For single feature
from dataset.VA_loader import load_data 
from configs.config_va import get_config

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchinfo import summary
import matplotlib.pyplot as plt
import torchmetrics.functional as tmf
import numpy as np
from tqdm import tqdm 


#The main pipeline of training
class Trainer():
    def __init__(self, train_loader, valid_loader):
        self.device = torch.device("cuda")
        #Dataloader
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.model = FF_model().to(self.device)
        #self.model = nn.DataParallel(self.model) 
        #summary(self.model, (1 , 128, 431))
        
        #Define optimizer and scheduler (schedule rule: half the lr if valid loss didn'nt decrease for two epoch)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=args.init_lr, weight_decay=args.l2_lambda)
        #self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, milestones=[2], gamma=1/60)
        #self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, args.epochs)
        #self.early_stop = args.early_stop
        self.count = 0
    
    def criterion(self, predict_va, label_va):
        #Define loss function
        MSE_loss = torch.nn.MSELoss()
        #Estimate total loss
        total_loss = MSE_loss(predict_va,label_va) 
        return total_loss

    @torch.no_grad()
    def valid_batch(self,batch):
        org_feat = batch[0].to(self.device)
        label_va = batch[1].to(self.device)
        predict_va = self.model(org_feat)
        #Estimate loss and R2 score and acc
        loss_batch = self.criterion(predict_va, label_va) 
        r2_valence = tmf.r2_score(predict_va[:, 0], label_va[:, 0])
        r2_arousal =  tmf.r2_score(predict_va[:, 1], label_va[:, 1])
        return loss_batch.item(), r2_valence.item(), r2_arousal.item()
    
    def valid_total(self):
        loss_total = 0.0
        r2_valence_total, r2_arousal_total = 0.0, 0.0
        for idx, batch in enumerate(tqdm(self.valid_loader, desc="Eval bar", colour="#9F35FF")):
            step = idx + 1
            loss_batch, r2_valence_batch, r2_arousal_batch = self.valid_batch(batch)        #Call for validating a batch
            #Accumalting r2_score and loss
            loss_total += loss_batch
            r2_valence_total += r2_valence_batch
            r2_arousal_total += r2_arousal_batch

        #Total loss & r2_score for the whole validation set
        loss_total = loss_total/step
        r2_valence_total = r2_valence_total/step
        r2_arousal_total = r2_arousal_total/step
        return loss_total, r2_valence_total, r2_arousal_total

    def train_batch(self, batch):
        org_feat = batch[0].to(self.device)
        label_va = batch[1].to(self.device)
        predict_va = self.model(org_feat)
        loss = self.criterion(predict_va, label_va)   #Estimate train loss
        self.optimizer.zero_grad()              #Clear the gradient in optimizer
        loss.backward()                         #Backward propogation
        self.optimizer.step()                   #Optimize
        return loss.item()

    def train_total(self):
        train_loss_list = []
        valid_loss_list = []
        r2_valence_list = []
        r2_arousal_list = []
        #min_val_loss = np.inf   #Initialize the minimum valid loss
        max_r2_sum = 0
        max_score_epoch = 0
        #valid_loss, r2_valence, r2_arousal = self.valid_total()
        for epoch in tqdm(range(args.epochs), desc="Epoch", colour="#0080FF"):
            self.model.train()  
            train_loss = 0.0
            for idx, batch in enumerate(tqdm(self.train_loader, desc=f"Train bar({epoch})", colour="#ff7300")):
                step = idx + 1
                loss_batch = self.train_batch(batch)        #Call for training a batch
                train_loss += loss_batch
            train_loss_list.append(train_loss/step)
            print(f"\n train loss: {train_loss/step}")

            self.model.eval()
            valid_loss, r2_valence, r2_arousal = self.valid_total()     #Validate every epoch after training
            r2_sum = r2_valence + r2_arousal
            valid_loss_list.append(valid_loss)
            r2_valence_list.append(r2_valence)
            r2_arousal_list.append(r2_arousal)
            #Show the valid loss and R2 score
            print(f"\n valid loss: {valid_loss} | R2_valence: {r2_valence} | R2_arousal: {r2_arousal}")     
            #Early stop
            if r2_sum > max_r2_sum:
                self.count = 0
                max_r2_sum = r2_sum
                max_score_epoch = epoch
                torch.save(self.model.state_dict(), args.model_save_path)
                print(f"Saving model..................................")
            else:
                self.count += 1
                #self.scheduler.step()    
            # if self.count >= self.early_stop:
            #         print("Early stopping...")
            #         break
  
        #Draw curve for R2 score versus epoch
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(r2_valence_list)), r2_valence_list, label='R2 valence', color='orange')
        plt.plot(range(len(r2_arousal_list)), r2_arousal_list, label='R2 arousal', color='purple')
        plt.xlabel('Epoch')
        plt.ylabel('R2')
        plt.title(f'R2 vs. Epoch best: {r2_valence_list[max_score_epoch]} | {r2_arousal_list[max_score_epoch]}')
        plt.legend()
        plt.grid(True)
        plt.savefig(args.curve_save_path+'/score.png')  
        plt.close()
        #Draw curve for loss versus epoch
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(train_loss_list)), train_loss_list, label='train_loss', color='blue')
        plt.plot(range(len(valid_loss_list)), valid_loss_list, label='valid_loss', color='red')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title(f'Loss vs. Epoch')
        # plt.title(f'Best ckpt: {np.argmin(valid_loss_list)} | {np.min(valid_loss_list)}')
        plt.legend()
        plt.grid(True)
        plt.savefig(args.curve_save_path+'/loss.png')   
        plt.close()

#-------------------------------------------------------------------------------------
def main(args):
    # Create dataloader
    train_loader, valid_loader = load_data(args.Trainset_Option, args.DEAM_datapath, args.Emomusic_datapath, args.PMemo_datapath, args.Mood_datapath, args.batch_size, 16)   
    #Set cuda
    trainer = Trainer(train_loader, valid_loader)
    trainer.train_total()

if __name__ == "__main__":
    args = get_config()     #Config from config.py 
    main(args)