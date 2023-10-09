import argparse
import os
import librosa
import numpy as np
import soundfile as sf
import torch
from tqdm import tqdm
from libft2gan.frame_transformer4 import FrameTransformerGenerator
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

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--gpu', '-g', type=int, default=-1)
    p.add_argument('--pretrained_model', '-P', type=str, default='model.pth') # https://mega.nz/file/P0xmGIKa#3_p23ugfOX3oSwgXlw7iayhMBxJNZ1HcZ-YgtbZToVw
    p.add_argument('--input', '-i', required=True)
    p.add_argument('--output', '-o', type=str, default="")
    p.add_argument('--outputformat', type=str, default="flac")
    p.add_argument('--num_res_encoders', type=int, default=4)
    p.add_argument('--num_res_decoders', type=int, default=4)
    p.add_argument('--sr', '-r', type=int, default=44100)
    p.add_argument('--n_fft', '-f', type=int, default=2048)
    p.add_argument('--hop_length', '-H', type=int, default=1024)
    p.add_argument('--batchsize', '-B', type=int, default=4)
    p.add_argument('--cropsize', '-c', type=int, default=1024)
    p.add_argument('--padding', type=int, default=512)
    p.add_argument('--output_image', '-I', action='store_true')
    p.add_argument('--postprocess', '-p', action='store_true')
    p.add_argument('--create_webm', action='store_true')
    p.add_argument('--create_vocals', action='store_true')
    p.add_argument('--num_encoders', type=int, default=2)
    p.add_argument('--num_decoders', type=int, default=13)
    p.add_argument('--tta', '-t', action='store_true')
    p.add_argument('--cropsizes', type=str, default='128,256,512,1024')
    p.add_argument('--depth', type=int, default=7)
    p.add_argument('--num_transformer_blocks', type=int, default=2)
    p.add_argument('--bias', type=str, default='true')
    p.add_argument('--rename_dir', type=str, default='false')
    
    p.add_argument('--num_attention_maps', type=int, default=2)
    p.add_argument('--channels', type=int, default=8)
    p.add_argument('--num_bridge_layers', type=int, default=4)
    p.add_argument('--latent_expansion', type=int, default=4)
    p.add_argument('--expansion', type=int, default=4)
    p.add_argument('--num_heads', type=int, default=8)
    p.add_argument('--dropout', type=float, default=0.35)
    p.add_argument('--weight_decay', type=float, default=1e-2)

    p.add_argument('--seed', type=int, default=1)

    p.add_argument('--num_res_blocks', type=int, default=1)
    p.add_argument('--feedforward_expansion', type=int, default=24)

    args = p.parse_args()
    args.cropsizes = [int(cropsize) for cropsize in args.cropsizes.split(',')]
    args.rename_dir = str.lower(args.rename_dir) == 'true'

    print('loading model...', end=' ')
    device = torch.device('cpu')

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    model = FrameTransformerGenerator(in_channels=2, out_channels=2, channels=args.channels, expansion=args.expansion, n_fft=args.n_fft, dropout=args.dropout, num_heads=args.num_heads, num_attention_maps=args.num_attention_maps, num_bridge_layers=args.num_bridge_layers, latent_expansion=args.latent_expansion)
    model.load_state_dict(torch.load(args.pretrained_model, map_location=device))

    if torch.cuda.is_available() and args.gpu >= 0:
        device = torch.device('cuda:{}'.format(args.gpu))
        model.to(device)
    print('done')

    output_folder = args.output
    outputformat = args.outputformat.lower()

    if outputformat not in ["flac", "wav", "mp3"]:
        outputformat = "flac"

    process = None
    queue = []

    if os.path.isdir(args.input):
        args.input = os.path.join(args.input, '')
        d = os.path.basename(os.path.dirname(args.input))
        output_folder = os.path.join(output_folder, d)
        print(output_folder)
        extensions = ['wav', 'm4a', 'mp3', 'mp4', 'flac', 'ogg']
        cover_ext = ["jpg", "png", "bmp"]

        cover = ""

        if output_folder != '' and not os.path.exists(output_folder):
            os.makedirs(output_folder)

        files = []
        d = os.listdir(args.input)
        for f in d:
            ext = f[::-1].split('.')[0][::-1]

            if ext in extensions:
                files.append(os.path.join(args.input, f))

            if ext in cover_ext:
                cover = os.path.join(args.input, f)

        for file in tqdm(files):
            print('loading wave source...', end=' ')
            X, sr = librosa.load(
                file, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
            basename = os.path.splitext(os.path.basename(file))[0]
            print('done')

            if X.ndim == 1:
                X = np.asarray([X, X])

            print('stft of wave source...', end=' ')
            X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
            print('done')

            sp = Separator(None, model, device, args.batchsize, args.cropsize, args.n_fft,   args.postprocess)

            if args.tta:
                y_spec, v_spec, m_spec = sp.separate_tta(X_spec)
            else:
                y_spec, v_spec, m_spec = sp.separate(X_spec, padding=args.padding)

            print('inverse stft of instruments...', end=' ')
            wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
            print('done')
            inst_file = '{}/{}_Instruments.{}'.format(output_folder, basename,outputformat)
            sf.write(inst_file, wave.T, sr)
            vid_file = '{}/{}.mp4'.format(output_folder, basename)


            if args.create_webm:
                os.system(f'ffmpeg -y -framerate 1 -loop 1 -i "{cover}" -i "{inst_file}" -t {librosa.get_duration(wave, sr=args.sr)} "{vid_file}"')

            if args.create_vocals:
                print('inverse stft of vocals...', end=' ')
                wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
                print('done')
                sf.write('{}/{}_Vocals.{}'.format(output_folder, basename, outputformat), wave.T, sr)
                
        if args.rename_dir:
            os.system(f'python song-renamer.py --dir "{output_folder}"')
            
    else:
        print('loading wave source...', end=' ')
        X, sr = librosa.load(
            args.input, args.sr, False, dtype=np.float32, res_type='kaiser_fast')
        basename = os.path.splitext(os.path.basename(args.input))[0]
        print('done')

        if X.ndim == 1:
            # mono to stereo
            X = np.asarray([X, X])

        print('stft of wave source...', end=' ')
        X_spec = spec_utils.wave_to_spectrogram(X, args.hop_length, args.n_fft)
        print('done')

        sp = Separator(None, model, device, args.batchsize, args.cropsize, args.n_fft, args.postprocess)

        if args.tta:
            y_spec, v_spec, m_spec = sp.separate_tta(X_spec)
        else:
            y_spec, v_spec, m_spec = sp.separate(X_spec, padding=args.padding)

        print('inverse stft of instruments...', end=' ')
        wave = spec_utils.spectrogram_to_wave(y_spec, hop_length=args.hop_length)
        print('done')
        sf.write('{}_Instruments.{}'.format(basename, outputformat), wave.T, sr)

        print('inverse stft of vocals...', end=' ')
        wave = spec_utils.spectrogram_to_wave(v_spec, hop_length=args.hop_length)
        print('done')
        sf.write('{}_Vocals.{}'.format(basename, outputformat), wave.T, sr)

        if args.output_image:
            image = spec_utils.spectrogram_to_image(y_spec)
            utils.imwrite('{}_Instruments.jpg'.format(basename), image)

            image = spec_utils.spectrogram_to_image(v_spec)
            utils.imwrite('{}_Vocals.jpg'.format(basename), image)

if __name__ == '__main__':
    main()
