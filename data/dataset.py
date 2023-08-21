import torch
import torchvision
from torch.utils.data.dataset import Dataset

import os, glob
from PIL import Image
import numpy as np

def open_txt(path):
    with open(path, 'r') as f:
        lst = [line.strip() for line in f.readlines()]
    return lst

def Unormalize(x):
    return (x * 255.0).type(torch.uint8)

class TherDataset(Dataset):
    def __init__(self,
                 root, txt_path,
                 img_resolution=[384, 512], si_resolution=[12, 16],
                 no_pair=False,
                 train=True):

        self.root = root
        self.no_pair = no_pair
        self.train = train
        
        lst = open_txt(txt_path)

        self.paths = [os.path.join(root, p) for p in lst]

        self.img_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(img_resolution, antialias=True)
        ])
        self.si_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(si_resolution, antialias=True)
        ])
        
        
    def __getitem__(self, index):
        img_path = source_path = self.paths[index]
        if self.no_pair:
            img_path = np.random.choice(self.paths)
        ID = img_path.replace(self.root, '').split('/')[1]

        # get wrong path
        while True:
            wrong_path = np.random.choice(self.paths)
            if wrong_path.replace(self.root, '').split('/')[1] != ID:
                break

        # get pair path
        while True:
            pair_path = np.random.choice(self.paths)
            if pair_path.replace(self.root, '').split('/')[1] == ID:
                break

        img_path = "{}.jpg".format(img_path)
        wrong_path = "{}.jpg".format(wrong_path)
        pair_path = "{}.jpg".format(pair_path)
        source_path = "{}.png".format(source_path)

        img = Image.open(img_path)
        wrong_img = Image.open(wrong_path)
        pair_img = Image.open(pair_path)
        img = self.img_transforms(img)
        wrong_img = self.img_transforms(wrong_img)
        pair_img = self.img_transforms(pair_img)

        source_img = Image.open(source_path)
        source_img = self.si_transforms(source_img)

        return img, source_img, pair_img, wrong_img

    def __len__(self):
        return len(self.paths)
    
import sys
sys.path.append('/root')
from TherAudio.data import audio
class TherSeqDataset(Dataset):
    def __init__(self,
                 root, txt_path, audio_root, audio_txt_path,
                 img_resolution=[384, 512], si_resolution=[12, 16],
                 no_pair=False,
                 sr=16000, fps=25, mel_step_size=16, T=5,
                 train=True):

        self.no_pair = no_pair
        self.train = train
        
        lst = open_txt(txt_path)
        self.paths = [os.path.join(root, p) for p in lst]

        self.img_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(img_resolution, antialias=True)
        ])
        self.si_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize(si_resolution, antialias=True)
        ])
        self.wav_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        # audio
        self.sr = sr
        self.mel_idx_multiplier = 80 / fps
        self.mel_step_size = mel_step_size
        self.T = T

        lst = open_txt(audio_txt_path)
        paths = [os.path.join(audio_root, p) for p in lst]
        self.audio_paths = []
        for p in paths:
            audio_paths = glob.glob(os.path.join(p, '*.wav'))
            self.audio_paths += audio_paths
        
    def read_wav(self, path):
        wav = audio.load_wav(path, self.sr)
        mel = audio.melspectrogram(wav)

        mel_chunks = []
        i = 0
        while True:
            start_idx = int(i * self.mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1
        return mel_chunks
    
    def read_frames(self, path, idx):
        frames = []
        for t in range(idx, idx+self.T):
            img_path = os.path.join(path, "{:03d}.png".format(t))
            img = Image.open(img_path)
            img = self.img_transforms(img)
            img = img.unsqueeze(0)
            frames.append(img)
        frames = torch.cat(frames, dim=0)
        return frames
        
    def __getitem__(self, index):
        ref_path = self.paths[index]
        source_path = self.paths[index]

        # get no pair data, ref != source
        if self.no_pair:
            ref_path = np.random.choice(self.paths)
        ref_path = "{}.jpg".format(ref_path)
        
        # get T source paths, T = 5
        source_paths = []
        number_str = source_path.split('/')[-1]
        number = int(number_str)
        for n in range(number, number+self.T):
            path = "{}.png".format(source_path.replace(number_str, "{:05d}".format(n)))
            source_paths.append(path)

        # read reference image from ref_path and transform
        ref = Image.open(ref_path)
        ref = self.img_transforms(ref)

        # read source images from source_paths and transform
        source_imgs = []
        for path in source_paths:
            source_img = Image.open(path)
            source_img = self.si_transforms(source_img)
            source_img = source_img.unsqueeze(0)
            source_imgs.append(source_img)
        source_imgs = torch.cat(source_imgs, dim=0)

        # random choose one .wav
        length = len(self.audio_paths)
        index = random.randint(0, length-1)
        wav_path = self.audio_paths[index]
        mel_chunks = self.read_wav(wav_path)

        # random choose one mel chunk and corresponding frames
        length = len(mel_chunks)
        start_idx = random.randrange(0, length-self.T)
        frames_folder_path = wav_path.split('.')[0]
        mel_frames = self.read_frames(frames_folder_path, start_idx)

        # get one mel from mel_chunk and convert to tensor
        mel = [self.wav_transforms(m).unsqueeze(0) for m in mel_chunks[start_idx: start_idx+self.T]]
        mel = torch.cat(mel, dim=0).type(torch.FloatTensor)

        return ref, source_imgs, mel, mel_frames

    def __len__(self):
        return len(self.paths)
    
import cv2
import random
class AudioDataset(Dataset):
    def __init__(self, root="/root/LRS2/mvlrs_v1/main", txt_path="/root/LRS2/trainlst.txt",
                 sr=16000, fps=25, mel_step_size=16, T=5,
                 face_size=96,
                 train=True,
                 face_model_path="/root/TherAudio/weights/yolov8n-face.onnx",
                 YTface_root="/root/YouTubeFaces/frame_images_DB", YTface_txt="/root/YouTubeFaces/trainList.txt",
                 expansion=16):
        lst = open_txt(txt_path)
        paths = [os.path.join(root, p) for p in lst]
        self.paths = []
        for p in paths:
            audio_paths = glob.glob(os.path.join(p, '*.wav'))
            self.paths += audio_paths

        self.YTface_root = YTface_root
        self.expansion = expansion
        YT_lst = open_txt(YTface_txt)
        self.YT_txt_path = [os.path.join(YTface_root, p)+".labeled_faces.txt" for p in YT_lst]

        self.sr = sr
        self.mel_idx_multiplier = 80 / fps
        self.mel_step_size = mel_step_size
        self.T = T
        self.face_size = face_size

        self.img_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize([face_size, face_size], antialias=True)
        ])
        self.wav_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])
        self.face_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize([size, size], antialias=True)
        ])

    def read_wav(self, path):
        wav = audio.load_wav(path, self.sr)
        mel = audio.melspectrogram(wav)

        mel_chunks = []
        i = 0
        while True:
            start_idx = int(i * self.mel_idx_multiplier)
            if start_idx + self.mel_step_size > len(mel[0]):
                mel_chunks.append(mel[:, len(mel[0]) - self.mel_step_size:])
                break
            mel_chunks.append(mel[:, start_idx: start_idx + self.mel_step_size])
            i += 1
        return mel_chunks

    def read_frames(self, path, idx):
        frames = []
        for t in range(idx, idx+self.T):
            img_path = os.path.join(path, "{:03d}.png".format(t))
            img = Image.open(img_path)
            img = self.img_transforms(img)
            # img = self.paste_on_zeros(img)
            img = img.unsqueeze(0)
            frames.append(img)
        frames = torch.cat(frames, dim=0)
        return frames
    
    def read_YT_frames(self, datas):
        frames = []
        faces = []
        for data in datas:
            path, _, x, y, w, h, _, _ = data.split(',')
            path = path.replace('\\', '/')
            path = os.path.join(self.YTface_root, path)

            x, y, w, h = [int(i) for i in [x, y, w, h]]

            img = Image.open(path)
            img = np.array(img)
            
            x1 = x-w//2 - self.expansion
            y1 = y-h//2 - self.expansion
            x2 = x+w//2 + self.expansion
            y2 = y+h//2 + self.expansion
            if x1 < 0: x1 = 0
            if y1 < 0: y1 = 0
            face = img[y1: y2, x1: x2, :]
            face = self.face_transforms(face)
            face = face.unsqueeze(0)
            faces.append(face)

        faces = torch.cat(faces, dim=0)
        return faces
    
    def get_start_idx(self, lst):
        length = len(lst)
        idx = random.randrange(0, length-self.T)
        w_idx = random.randrange(0, length-self.T)
        while w_idx == idx:
            w_idx = random.randrange(0, length-self.T)
        return idx, w_idx

    def __getitem__(self, index):
        wav_path = self.paths[index]
        mel_chunks = self.read_wav(wav_path)

        start_idx, wrong_idx = self.get_start_idx(mel_chunks)
        frames_folder_path = wav_path.split('.')[0]

        frames = self.read_frames(frames_folder_path, start_idx)
        gt = frames.clone()
        wrong_frames = self.read_frames(frames_folder_path, wrong_idx)

        frames[:, :, self.face_size//2:, :] = 0 # mouth part to 0

        mel = [self.wav_transforms(m).unsqueeze(0) for m in mel_chunks[start_idx: start_idx+self.T]]
        mel = torch.cat(mel, dim=0).type(torch.FloatTensor)
        mel_forSync = mel[0]

        input = torch.cat([frames, wrong_frames], dim=1)
        
        return input, gt, mel, mel_forSync

    def __len__(self):
        return len(self.paths)
    
class BorderDataset(Dataset):
    def __init__(self, root="/root/YouTubeFaces/frame_images_DB", txt_path="/root/YouTubeFaces/trainList.txt", return_img=False, face_size=96):
        super().__init__()
        self.root = root
        self.return_img = return_img

        lst = open_txt(txt_path)
        paths = [os.path.join(root, p)+".labeled_faces.txt" for p in lst]
        self.folders = []
        for p in paths:
            txt = open_txt(p)
            self.folders.append(txt)

        self.img_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
        ])

        self.face_transforms = torchvision.transforms.Compose([
            torchvision.transforms.Resize([face_size, face_size], antialias=True)
        ])

    def split_data(self, data):
        path, _, x, y, w, h, _, _ = data.split(',')
        path = path.replace('\\', '/')
        path = os.path.join(self.root, path)

        x, y, w, h = [int(i) for i in [x, y, w, h]]
        
        return path, x, y, w, h
    
    def read_and_crop(self, path, x, y, w, h):
        img = Image.open(path)
        img = np.array(img)
        img = self.img_transforms(img)
        
        x1 = x - w//2
        y1 = y - h//2
        x2 = x + w//2
        y2 = y + h//2
        if x1 < 0: x1 = 0
        if y1 < 0: y1 = 0
        face = img[:, y1: y2, x1: x2]
        face = self.face_transforms(face)

        return img, face
        

    def __getitem__(self, index):
        folder = self.folders[index]
        length = len(folder)

        # get ref data
        ref_idx = random.randrange(0, length-1)
        ref_data = folder[ref_idx]
        ref_path, ref_x, ref_y, ref_w, ref_h = self.split_data(ref_data)
        ref_video_id = ref_path.split('/')[-2]

        # get target data
        while True:
            target_idx = random.randrange(0, length-1)
            target_data = folder[target_idx]
            target_path, target_x, target_y, target_w, target_h = self.split_data(target_data)
            target_video_id = target_path.split('/')[-2]
            if target_idx != ref_idx and target_video_id == ref_video_id:
                break
        
        ref_img, ref_face = self.read_and_crop(ref_path, ref_x, ref_y, ref_w, ref_h)
        target_img, target_face = self.read_and_crop(target_path, target_x, target_y, target_w, target_h)

        if self.return_img:
            return [ref_img, ref_face], [target_img, target_face], [target_x, target_y, target_w, target_h]
        else:
            return ref_face, target_face

    def __len__(self):
        return len(self.folders)


if __name__ == '__main__':
    # dataset = TherDataset(root="/root/thermal_data", txt_path="/root/thermal_data/trainList.txt",
    #                       no_pair=True)
    # dataset = TherSeqDataset(root="/root/thermal_data", txt_path="/root/thermal_data/trainList.txt",
    #                          audio_root="/root/LRS2/mvlrs_v1/main", audio_txt_path="/root/LRS2/vallst.txt",
    #                          no_pair=True, train=False)
    # dataset = AudioDataset()
    dataset = BorderDataset()
    for d in dataset:
        for i in d:
            print(i.shape)
        break
        pass
        # input, gt, mel, mel_forSync, frames_forSync, y_forSync = d
        # print(input.shape, gt.shape, mel.shape, mel_forSync.shape, frames_forSync.shape)
        # print(y_forSync)
        # break