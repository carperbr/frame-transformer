import argparse
import os
import re
import shutil
import librosa
import numpy as np
import soundfile as sf
import music_tag
import torch
from tqdm import tqdm
from libft2gan.frame_transformer5 import FrameTransformer as FrameTransformerV8
from libft2gan_v7.frame_transformer4 import FrameTransformerGenerator as FrameTransformerV7
from lib import dataset
from lib import spec_utils
from lib import utils

class Separator(object):
    def __init__(self, corrector, model, device, batchsize, cropsize, n_fft, postprocess=False):
        self.corrector = corrector
        self.model = model
        self.offset = 0
        self.device = device
        self.batchsize = batchsize
        self.cropsize = cropsize
        self.postprocess = postprocess
        self.n_fft = n_fft

    def _separate(self, X_mag_pad, cropsize=None, padding=None):
        X_dataset = []
        cropsize = self.cropsize if cropsize is None else cropsize
        padding = cropsize // 2 if padding is None else padding
        patches = X_mag_pad.shape[2] // cropsize
        X_mag_pad = np.pad(X_mag_pad, ((0, 0), (0, 0), (padding, padding)), mode='constant')
        for i in range(patches):
            start = (i * cropsize) + padding
            X_mag_crop = X_mag_pad[:, :, (start - padding):(start + cropsize + padding)]
            X_dataset.append(X_mag_crop)

        self.model.eval()
        with torch.no_grad():
            mask = []
            # To reduce the overhead, dataloader is not used.
            for i in tqdm(range(0, patches, self.batchsize)):
                X_batch = X_dataset[i: i + self.batchsize]
                X_batch = torch.from_numpy(np.asarray(X_batch)).to(self.device)[:, :, :(self.n_fft // 2)]

                with torch.cuda.amp.autocast_mode.autocast(enabled=True):
                    pred = torch.sigmoid(self.model(X_batch))

                if padding > 0:
                    pred = pred[:, :, :, (padding):-(padding)]

                pred = pred.detach().cpu().numpy()
                pred = np.concatenate(pred, axis=2)
                mask.append(pred)

            mask = np.concatenate(mask, axis=2)

        mask = np.pad(mask, ((0,0), (0,1), (0, 0)))

        return mask

    def _preprocess(self, X_spec):
        X_mag = np.abs(X_spec)
        X_phase = np.angle(X_spec)

        return X_mag, X_phase

    def _postprocess(self, mask, X_mag, X_phase):
        if self.postprocess:
            mask = spec_utils.merge_artifacts(mask)

        y_spec = mask * X_mag * np.exp(1.j * X_phase)
        v_spec = (1 - mask) * X_mag * np.exp(1.j * X_phase)
        m_spec = mask * 255

        return y_spec, v_spec, m_spec

    def separate(self, X_spec, padding=None):
        X_mag, X_phase = self._preprocess(X_spec)
        n_frame = X_mag.shape[2]
        pad_l, pad_r, _ = dataset.make_padding(n_frame, self.cropsize, 0)
        xm = X_mag / X_mag.max()
        X_mag_pad = np.pad(xm, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
        mask = self._separate(X_mag_pad, self.cropsize, padding)

        mask = mask[:, :, :n_frame]
        y_spec, v_spec, m_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec, m_spec

    def separate_tta(self, X_spec, cropsizes=[64, 128, 256, 512, 1024], paddings=[128, 256, 512, 1024, 2048]):
        X_mag, X_phase = self._preprocess(X_spec)

        n_frame = X_mag.shape[2]
        X_mag_pad1 = X_mag / X_mag.max()

        mask = np.zeros_like(X_mag)

        for idx in range(len(paddings)):
            pad_l, pad_r, _ = dataset.make_padding(n_frame, paddings[idx], 0)
            X_mag_pad2 = np.pad(X_mag_pad1, ((0, 0), (0, 0), (pad_l, pad_r)), mode='constant')
            mask += self._separate(X_mag_pad2, cropsizes[idx], paddings[idx])[:, :, :n_frame]

        mask = mask / len(paddings)

        y_spec, v_spec, m_spec = self._postprocess(mask, X_mag, X_phase)

        return y_spec, v_spec, m_spec

def parse_arguments(p):
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model_v8', type=str, default='model.v8.1.pth') # https://mega.nz/file/n8IUwJZK#D9-1u7RqYZF4lt_lJ5By58PPj4a0itdojF-fUZf_WQA
    p.add_argument('--pretrained_model_v7', type=str, default='model.v7.2.pth') # https://mega.nz/file/L55UwZAC#leWuP1ic9Rt4SpDVHeo6tnimaEIw5095ORoiaosF1Do
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--output', '-o', type=str, default="")
    p.add_argument('--output_format', type=str, default="flac")
    p.add_argument('--num_res_encoders', type=int, default=4)
    p.add_argument('--num_res_decoders', type=int, default=4)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--cropsize', '-c', type=int, default=1024)
    p.add_argument('--padding', type=int, default=512)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--copy_source_images', action='store_true') # copies images from input into output
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--path_from_tags', action='store_true') # names final folder from tags rather than original
    p.add_argument('--create_webm', action='store_true')
    p.add_argument('--create_vocals', action='store_true')
    p.add_argument('--num_encoders', type=int, default=2)
    p.add_argument('--num_decoders', type=int, default=13)
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--cropsizes', type=str, default='128,256,512,1024')
    p.add_argument('--depth', type=int, default=7)
    p.add_argument('--num_transformer_blocks', type=int, default=2)
    p.add_argument('--bias', type=str, default='true')
    p.add_argument('--num_attention_maps', type=int, default=1)
    p.add_argument('--channels', type=int, default=8)
    p.add_argument('--num_bridge_layers', type=int, default=4)
    p.add_argument('--latent_expansion', type=int, default=4)
    p.add_argument('--expansion', type=float, default=2.2)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.2)
    p.add_argument('--weight_decay', type=float, default=1e-2)
    p.add_argument('--seed', type=int, default=1)
    p.add_argument('--num_res_blocks', type=int, default=1)
    p.add_argument('--feedforward_expansion', type=int, default=24)
    # p.add_argument('--rename_dir', action='store_true')

    return p.parse_args()

def load_models(args):
    # print('loading model v7...', end=' ')
    device = torch.device('cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model_v7 = FrameTransformerV7(in_channels=2, out_channels=2, channels=8, dropout=0, n_fft=2048, num_heads=8, expansion=4, latent_expansion=4, num_attention_maps=2)
    model_v7.load_state_dict(torch.load(args.pretrained_model_v7))
    
    # print('loading model v8...', end=' ')

    model_v8 = FrameTransformerV8(in_channels=2, out_channels=2, channels=8, expansion=2.2, n_fft=2048, dropout=0, num_heads=8, num_attention_maps=1)
    model_v8.load_state_dict(torch.load(args.pretrained_model_v8))

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        # model_v8.to(device)
    # print('done')

    return model_v7, model_v8, device

def transform_track(args, file, fullpath):
    
    model_v7, model_v8, device = load_models(args)

    output_format = args.output_format

    print('\nloading wave source...', end=' ')
    X, sr = librosa.load(
        file, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
    basename = os.path.splitext(os.path.basename(file))[0]
    print(basename)

    if X.ndim == 1:
        X = np.asarray([X, X])

    print('stft of wave source...', end=' ')
    X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
    print('done')

    model_v8.to(device)
    sp = Separator(None, model_v8, device, args.batchsize, args.cropsize, args.n_fft,   args.postprocess)

    if args.tta:
        y_spec_a, v_spec_a, _ = sp.separate_tta(X_spec)
    else:
        y_spec_a, v_spec_a, _ = sp.separate(X_spec, padding=args.padding)

    model_v8.to('cpu')
    model_v7.to(device)

    sp = Separator(None, model_v7, device, args.batchsize, args.cropsize, args.n_fft,   args.postprocess)

    if args.tta:
        y_spec_b, v_spec_b, _ = sp.separate_tta(X_spec)
    else:
        y_spec_b, v_spec_b, _ = sp.separate(X_spec, padding=args.padding)

    y_mag_a = np.abs(y_spec_a)
    y_mag_b = np.abs(y_spec_b)
    v_mag_a = np.abs(v_spec_a)
    v_mag_b = np.abs(v_spec_b)
    y_phase = np.angle(y_spec_a)
    v_phase = np.angle(v_spec_a)
    y_mag = np.minimum(y_mag_a, y_mag_b)
    v_mag = np.maximum(v_mag_a, v_mag_b)
    y_spec = y_mag * np.exp(1.j * y_phase)
    v_spec = v_mag * np.exp(1.j * v_phase)

    print('\ninverse stft of instruments...', end=' ')
    wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
    print('done')
    inst_file = f'{fullpath}/{basename}_Instruments.{output_format}'
    sf.write(inst_file, wave.T, sr)

    copy_tags(file,inst_file,"instruments")

    # if args.create_webm:
    #     vid_file = f'{fullpath}/{basename}.mp4'
    #     os.system(f'ffmpeg -y -framerate 1 -loop 1 -i "{coverfiles[0]}" -i "{inst_file}" -t {librosa.get_duration(wave, sr=args.sr)} "{vid_file}"')

    if args.create_vocals:
        print('inverse stft of vocals...', end=' ')
        wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
        print('done')
        voc_file = f'{fullpath}/{basename}_Vocals.{output_format}'
        sf.write(voc_file, wave.T, sr)

    copy_tags(file,voc_file,"vocals")

def copy_tags(input, output, suffix):
    #input ~ 'C://input/tagfile.mp3'
    #output ~ 'H://output/tagfile_Instruments.flac'
    #suffix ~ 'instruments'

    suffix = f'({suffix.capitalize()})' # >> '(Instruments)'

    input = music_tag.load_file(input)
    output = music_tag.load_file(output)
    output['tracktitle'] = f'{input["tracktitle"]} {suffix}'
    output['album'] = f'{input["album"]} {suffix}'
    output['artist'] = input['artist']
    output['year'] = input['year']
    try:
        output['tracknumber'] = input['tracknumber'] 
        #NAN error if, for example, the tracknumber is stored as "1/10"
    except:
        output['tracknumber'] = re.sub('^([0-9]+).*$', "\\1", (input.raw['tracknumber'].value)) 
        # takes the first set of digits in the raw tracknumber

    try:
        output['totaltracks'] = input['totaltracks']
    except:
        output['totaltracks'] = re.sub('^[0-9]+.*([0-9])+$', "\\1", (input.raw['tracknumber'].value)) 
        # takes the second set of digits in the raw tracknumber

    output.save()

def basedirectory_from_tags(file:str):
    filetag =  music_tag.load_file(file)
    basedirectory = f'{filetag["artist"]} - {filetag["album"]}'
    # print(out)
    return basedirectory

def create_outputdirectory(root, basedirectory):
    fulldirectory = os.path.join(root, basedirectory)
    if fulldirectory != '' and not os.path.exists(fulldirectory):
        os.makedirs(fulldirectory)
    return fulldirectory


def natural_sort(l): 
    #https://stackoverflow.com/a/4836734
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [convert(c) for c in re.split('([0-9]+)', key)] 
    return sorted(l, key=alphanum_key)

def main():
    args = parse_arguments(argparse.ArgumentParser())
    
    args.cropsizes = [int(cropsize) for cropsize in args.cropsizes.split(',')]
    # args.rename_dir = str.lower(args.rename_dir) == 'true'

    output_folder = args.output
    if output_folder != '' and not os.path.exists(output_folder):
        os.makedirs(output_folder)

    output_format = args.output_format.lower()
    if output_format not in ["flac", "wav", "mp3"]:
        output_format = "flac"

    args.output_format = output_format

    if os.path.isdir(args.input):
        args.input = os.path.join(args.input, '')

        files = []
        coverfiles = []
        sound_ext = ['wav', 'm4a', 'mp3', 'mp4', 'flac', 'ogg']
        cover_ext = ["jpg", "png", "bmp"]     
        directoryfiles = natural_sort(os.listdir(args.input))
        
        for f in directoryfiles:
            ext = f[::-1].split('.')[0][::-1]

            if ext in sound_ext:
                files.append(os.path.join(args.input, f))                

            if ext in cover_ext:
                coverfiles.append(os.path.join(args.input, f))
        
        fullpath = ''        

        if args.path_from_tags:
            firstsong = os.path.join(args.input, directoryfiles[0])
            fullpath = create_outputdirectory(output_folder, basedirectory_from_tags(firstsong))        
        else:
            d = os.path.basename(os.path.dirname(args.input))
            fullpath = create_outputdirectory(output_folder, d)

        print(fullpath)

        if args.copy_source_images:
            for f in coverfiles:
                shutil.copy(f, fullpath)	

        for file in tqdm(files):
            transform_track(args, file, fullpath)
                        
    else: # args.input is a single file
            transform_track(args, args.input, output_folder)
        
if __name__ == '__main__':
    main()
