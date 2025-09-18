from model.linear_va import FF_model
from openai import OpenAI
import argparse
import os
import numpy as np 
import torch
import csv
import matplotlib.pyplot as plt
from tqdm import tqdm




def plot_testdata_gpt(args):
    #Model setup
    client = OpenAI(api_key=args.api_key) #OpenAI GPT setup
    model = FF_model().to('cuda')
    model.load_state_dict(torch.load(args.model_path,weights_only=True))
    model.eval()

    #The function for calling GPT api with prompt
    def GPT_response(lyrics_path):
        with open(lyrics_path) as f:
            lyrics = f.read()
        prompt = f"""
        You are a music emotion classifier.
        Task: Determine whether the following song lyrics express a positive (valence > 0) 
        or negative (valence < 0) emotional valence.
        Lyrics:
        {lyrics}
        Answer format: Output only one number, "1" for positive valence, or "0" for negative valence.
        """
        response = client.responses.create(model="gpt-5",input=prompt)
        return response.output_text.strip().lower()

    #Data prepare
    datapath = args.feature_path
    lyrics_path = args.lyrics_path
    song_list = os.listdir(datapath)
    song_id, id_map = 0, []
    pred_list = []

    #Get song id v.s song name && song id on the VA plot
    for song in tqdm(song_list):
        id_map.append({"song_id": song_id, "song_name": os.path.splitext(song)[0] })
        song_id += 1
        #load preprocessed feature and label
        org_feature_stack= np.load(os.path.join(datapath,song)) 
        org_features = org_feature_stack.mean(axis=0)
        #Inference
        with torch.no_grad():
            org_input = torch.tensor(org_features).unsqueeze(0).to('cuda')
            predict_va = model(org_input).squeeze(0).cpu()
        #Check if the sign of valence from LLM == model
        valence_sign = int(GPT_response(os.path.join(lyrics_path,song.replace(".npy",".txt"))))
        #Reflect: 5 as the middle
        if (valence_sign==1)and(predict_va[0]<5):
            predict_va[0] = -predict_va[0]+10
        elif (valence_sign==0)and(predict_va[0]>5):
            predict_va[0] = -predict_va[0]+10      
        #Save the prediction to a list 
        pred_list.append(predict_va)

    # Save song id vs song name as .csv
    csv_path = "./song_mapping.csv"
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["song_id", "song_name"])
        writer.writeheader()
        writer.writerows(id_map)

    #Plot VA plane
    pred_array = np.stack(pred_list)  # shape: (N, 2)，每列 [valence, arousal]
    valence,arousal = pred_array[:, 0], pred_array[:, 1]
    plt.figure(figsize=(8, 8))
    plt.scatter(valence, arousal, c="blue", alpha=0.6)

    #  Mark the ID under every point
    for idx, (v, a) in enumerate(zip(valence, arousal)):
        plt.text(v, a - 0.15, str(idx), fontsize=8, ha="center", va="top", color="red")

    # Plot Setting 
    plt.xlim(1, 9)
    plt.ylim(1, 9)
    plt.xticks(range(1, 10))
    plt.yticks(range(1, 10))
    plt.axhline(5, color="gray", linestyle="--", linewidth=0.8)  # y=5 參考線
    plt.axvline(5, color="gray", linestyle="--", linewidth=0.8)  # x=5 參考線
    plt.xlabel("Valence")
    plt.ylabel("Arousal")
    plt.title("Predicted Valence-Arousal Scatter Plot (1~9 scale)")
    plt.grid(True, linestyle="--", alpha=0.5)
    # plt.savefig("./plot/M_only_GPT.png", dpi=300)

#--------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--api_key', type=str, default='xxxxxxxxxxxxxxxx', help="Please fill in you OpenAI api key")
    parser.add_argument('--feature_path', type=str, default='/mnt/gestalt/home/dcn2001/xrspace_testdata/feature_5s', help="Path of the extracted testdata feature")
    parser.add_argument('--lyrics_path', type=str, default='/mnt/gestalt/home/dcn2001/xrspace_testdata/lyrics_sample', help="Path of the transccripted testdata lyrics")
    parser.add_argument('--model_path', type=str, default='./model_state/M_only.pth', help="Path of the model state")
    parser.add_argument('--plot_path', type=str, default='./plot/M_only.png', help="The path you want to save the plot")
    args = parser.parse_args()
    
    plot_testdata_gpt(args)
