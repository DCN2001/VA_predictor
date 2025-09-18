import argparse
import os
import demucs.separate
import shutil
from tqdm import tqdm

#Select the file name in feature folder and read from audio folder 
def get_vocal(args):
    device = 'cuda'
    #Start separation
    file_list = os.listdir(args.audio_dir)
    for file in tqdm(file_list,colour="red"):  
        song_path = os.path.join(args.audio_dir,file)
        demucs.separate.main(['-d', device,'--mp3-preset', '2','-o', args.vocal_dir,'--filename', file, song_path])
    #Handle the bug of DEMUCS move from htdemucs folder to dest folder and remove htdemucs folder 
    htdemucs_folder = os.path.join(args.vocal_dir,"htdemucs")
    htdemucs_list = os.listdir(htdemucs_folder)
    for file in htdemucs_list:
        src, dest = os.path.join(htdemucs_folder,file), os.path.join(args.vocal_dir,file)
        shutil.move(src,dest)
    shutil.rmtree(htdemucs_folder)  #Remove HTdemucs folder

#------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--audio_dir', type=str, default='/mnt/gestalt/home/dcn2001/xrspace_testdata/sample',help="Paht of your original audio")
    parser.add_argument('--vocal_dir', type=str, default='/mnt/gestalt/home/dcn2001/xrspace_testdata/vocal_audio_sample',help="Path of saving your vocal part audio")
    args = parser.parse_args()
    
    get_vocal(args)