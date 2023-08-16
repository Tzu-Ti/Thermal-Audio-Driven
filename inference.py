__author__ = 'Titi'
from multiprocessing import cpu_count
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import torch
from torch.utils.data import DataLoader
from torch.nn import DataParallel
import torchvision.transforms.functional as TF

from train import Model_factory as SPADE_factory
from train_face import Model_factory as Face_factory
from data.dataset import TherDataset, TherSeqDataset, Unormalize
from utils import detectFace

import argparse, os
import einops
import face_alignment

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # training setting
    parser.add_argument('--model_name', default='test')
    parser.add_argument('--batch_size', type=int, default=4)
    # Model setting
    parser.add_argument('--filter', type=int, default=32)
    parser.add_argument('--z_dim', type=int, default=256)
    # I/O
    parser.add_argument('--root', default='/root/thermal_data')
    parser.add_argument('--val_txt', default='/root/thermal_data/valList.txt')
    parser.add_argument('--test_txt', default='/root/thermal_data/testList.txt')
    parser.add_argument('--audio_root', default='/root/LRS2/mvlrs_v1/main')
    parser.add_argument('--audio_txt_path', default="/root/LRS2/vallst.txt")
    # loss weight
    parser.add_argument('--w_kld', type=float, default=0.05)
    parser.add_argument('--w_feat', type=float, default=10.0)
    parser.add_argument('--w_vgg', type=float, default=10.0)
    parser.add_argument('--w_rec', type=float, default=10.0)
    # face
    parser.add_argument('--face_model_path', default='weights/yolov8n-face.onnx')
    parser.add_argument('--T', type=int, default=5)
    parser.add_argument('--face_size', type=int, default=96)
    # checkpoint setting
    parser.add_argument('--spade_ckpt_path', default='lightning_logs/0728/FirstTry/checkpoints/epoch=29-step=225000.ckpt')
    parser.add_argument('--lip_ckpt_path', default='lightning_logs/0815/CorrectData/checkpoints/epoch=129-step=77480.ckpt')
    parser.add_argument('--sync_weight_path', default='weights/lipsync_expert.pth')

    return parser.parse_args()

class Model_factory(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        # load face detector and convert to DataParallel
        self.FaceDetector = face_alignment.FaceAlignment(face_alignment.LandmarksType.TWO_D, device='cuda')
        self.FaceDetector.face_detector.face_detector = DataParallel(self.FaceDetector.face_detector.face_detector, device_ids=[0, 1])
        # load SPADE model
        self.SPADE_model = SPADE_factory.load_from_checkpoint(args.spade_ckpt_path, args=args)
        self.SPADE_model.eval()
        # load lip model
        self.lip_model = Face_factory.load_from_checkpoint(args.lip_ckpt_path, args=args)
        self.lip_model.eval()

    def audio_drive_ref(self, img, audio):
        _, batch_ref_face = detectFace(detector=self.FaceDetector, imgs=img, size=self.args.face_size)
        # repeat ref face for T times
        batch_ref_seq_face = batch_ref_face.unsqueeze(1)
        batch_ref_seq_face = einops.repeat(batch_ref_seq_face, 'B 1 C H W -> B T C H W', T=self.args.T)
        # set the bottom half of the face to 0
        B, T, C, H, W = batch_ref_seq_face.shape
        batch_ref_seq_half_face = batch_ref_seq_face.clone()
        batch_ref_seq_half_face[:, :, :, H//2:, :] = 0

        # concat half face and full face
        audio_input = torch.cat([batch_ref_seq_half_face, batch_ref_seq_face], dim=2)

        # generate talking face
        talking_face = self.lip_model.netG(audio, audio_input)

        return talking_face
    
    def face_swap(self, fake_images, talking_face):
        # fake_images, talking_face: B, T, C, H, W
        # detect fake_images face
        coords = []
        fake_images = fake_images.clone()
        new_fake_images = torch.zeros_like(fake_images)
        for t in range(self.args.T):
            batch_talking_face = talking_face[:, t]
            batch_fake_images = fake_images[:, t]
            # iterative detect face on T. batch_face_coords: B, [x1, y1, x2, y2]. batch_fake_face: B, C, H, W. 
            batch_face_coords, batch_fake_face = detectFace(detector=self.FaceDetector, imgs=fake_images[:, t], size=self.args.face_size)
            
            # iterative on batch
            for idx, (coords, face, fake_image) in enumerate(zip(batch_face_coords, batch_talking_face, batch_fake_images)):
                # crop talking face
                x1, y1, x2, y2 = coords
                h = y2 - y1
                w = x2 - x1
                face = TF.resize(face, (h, w))
                fake_image[:, y1:y2, x1:x2] = face
                new_fake_images[idx, t] = fake_image
        return new_fake_images
    
    def predict_step(self, batch, batch_idx):
        ref, source_imgs, mel, mel_frames = batch
        self.source_imgs = source_imgs
        self.mel_frames = mel_frames
        # copy ref for T times, B, T, C, H, W
        refs = torch.cat([ref.unsqueeze(1) for i in range(self.args.T)], dim=1)
        self.imgs = refs
        
        # rearrange ref and source_imgs to B*T, C, H, W
        refs = einops.rearrange(refs, 'B T C H W -> (B T) C H W')
        source_imgs = einops.rearrange(source_imgs, 'B T C H W -> (B T) C H W')

        # generate RGB images from heatmaps
        fake_images, _ = self.SPADE_model.generate(refs, source_imgs)
       
        # rearrange fake_images back to B, T, C, H, W
        fake_images = einops.rearrange(fake_images, '(B T) C H W -> B T C H W', T=self.args.T)
        self.fake_images = fake_images
        
        # generate talking face from ref image. B, T, C, H, W
        talking_face = self.audio_drive_ref(ref, mel)
        self.talking_face = talking_face

        # face swap
        try:
            # new_fake_images = torch.zeros_like(fake_images)
            new_fake_images = self.face_swap(fake_images, talking_face)
        except:
            new_fake_images = torch.zeros_like(fake_images)
            print("face swap error")
        self.new_fake_images = new_fake_images

    def on_predict_end(self):
        self.logger.experiment.add_images("Predict/img", Unormalize(self.imgs[0]), self.current_epoch)
        self.logger.experiment.add_images("Predict/source_image", Unormalize(self.source_imgs[0]), self.current_epoch)
        self.logger.experiment.add_images("Predict/fake_image", Unormalize(self.fake_images[0]), self.current_epoch)
        self.logger.experiment.add_images("Predict/talking_face", Unormalize(self.talking_face[0]), self.current_epoch)
        self.logger.experiment.add_images("Predict/new_fake_image", Unormalize(self.new_fake_images[0]), self.current_epoch)
        self.logger.experiment.add_images("Predict/talking_face_gt", Unormalize(self.mel_frames[0]), self.current_epoch)
        
def main():
    args = parse()

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.getcwd(),
        version=args.model_name,
        name='lightning_logs'
    )
    
    testDataset = TherSeqDataset(root=args.root, txt_path=args.val_txt,
                                 audio_root=args.audio_root, audio_txt_path=args.audio_txt_path,
                                 no_pair=True, train=False)
    testDataloader = DataLoader(dataset=testDataset,
                                batch_size=args.batch_size,
                                shuffle=True, 
                                num_workers=cpu_count())
    
    Model = Model_factory(args)
    trainer = pl.Trainer(logger=tb_logger)
    trainer.predict(Model, dataloaders=testDataloader)

if __name__ == '__main__':
    main()