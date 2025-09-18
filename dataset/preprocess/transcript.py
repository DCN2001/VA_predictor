import argparse
import os
import whisper
from tqdm import tqdm


def get_lyric(args):
    audio_folder = args.vocal_dir
    lyric_folder = args.lyric_dir
    model = whisper.load_model('large-v3', device="cuda")

    song_list = os.listdir(audio_folder)
    for song in tqdm(song_list):
        song_path = os.path.join(audio_folder,song)
        transcript = model.transcribe(song_path)
        with open(os.path.join(lyric_folder, f'{os.path.splitext(os.path.split(song)[-1])[0]}.txt'), 'w') as f:
            for segment in transcript['segments']:
                f.write(f"{segment['text'].strip()}\n")

#------------------------------------------------------------------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vocal_dir', type=str, default='/home/dcn2001/NTU_NAS1/xrspace_testdata/vocal_audio_sample',help="Path of your vocal audio")
    parser.add_argument('--lyric_dir', type=str, default='/home/dcn2001/NTU_NAS1/xrspace_testdata/lyrics_sample',help="Path of the transcription")
    args = parser.parse_args()
    
    get_lyric(args)