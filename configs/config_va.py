import argparse

#Config for training model
def get_config():
    parser = argparse.ArgumentParser()
    #Train params
    parser.add_argument("--epochs", type=int, default=150, help="number of epochs")
    parser.add_argument("--batch_size", type=int, default=64, help="number per batch")
    parser.add_argument("--init_lr", type=float, default=1e-4)   
    parser.add_argument("--l2_lambda", type=float, default=1e-2)
    # parser.add_argument('--early_stop', type=int, default=5)
    #Dataset
    parser.add_argument("--Trainset_Option", type=int, default="M_only", help="DEP_only or M_only or DEPM")
    parser.add_argument("--DEAM_datapath", type=str, default="/mnt/gestalt/home/dcn2001/DEAM_preprocessed")
    parser.add_argument("--Emomusic_datapath", type=str, default="/mnt/gestalt/home/dcn2001/Emomusic_preprocessed")
    parser.add_argument("--PMemo_datapath", type=str, default="/mnt/gestalt/home/dcn2001/PMemo_preprocessed")
    parser.add_argument("--Mood_datapath", type=str, default="/mnt/gestalt/home/dcn2001/Mood_preprocessed")
    #Save path
    parser.add_argument("--model_save_path", type=str, default="./model_state/DEPM.pth")
    parser.add_argument("--curve_save_path", type=str, default="./curve",help="The path of 'epoch v.s metric' & 'epoch v.s loss'")
    args = parser.parse_args()
    return args