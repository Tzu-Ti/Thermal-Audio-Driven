__author__ = 'Titi'
from multiprocessing import cpu_count
from typing import Any
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers

import torch
from torch.utils.data import DataLoader

from train import Model_factory as SPADE_factory
from train import parse as SPADE_parse
from data.dataset import TherDataset, TherSeqDataset, Unormalize
from face import YOLOv8_face
from utils import detectFace

import argparse, os
import einops

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
    # visualization
    parser.add_argument('--show_num', type=int, default=4)
    # checkpoint setting
    parser.add_argument('--spade_ckpt_path', default='lightning_logs/0728/FirstTry/checkpoints/epoch=29-step=225000.ckpt')

    return parser.parse_args()

class Model_factory(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.FaceDetector = YOLOv8_face(self.args.face_model_path, conf_thres=0.45, iou_thres=0.5)
        self.SPADE_model = SPADE_factory.load_from_checkpoint(args.spade_ckpt_path, args=args)

    def audio_drive_ref(self, img, audio):
        coord, ref_face = detectFace(detector=self.FaceDetector, img=img.squeeze(0))
        print(coord)
    
    def predict_step(self, batch, batch_idx):
        ref, source_imgs, mel, mel_frames = batch
        # copy ref for T times, B, T, C, H, W
        refs = torch.cat([ref.unsqueeze(1) for i in range(self.args.T)], dim=1)
        
        # rearrange ref and source_imgs to B*T, C, H, W
        refs = einops.rearrange(refs, 'B T C H W -> (B T) C H W')
        source_imgs = einops.rearrange(source_imgs, 'B T C H W -> (B T) C H W')

        # generate RGB images from heatmaps
        fake_images, _ = self.SPADE_model.generate(refs, source_imgs)

        # rearrange fake_images back to B, T, C, H, W
        print(fake_images.shape)
        raise
        # generate talking face from ref image
        self.audio_drive_ref(img, None)

        self.imgs = imgs
        self.source_imgs = source_imgs
        self.fake_images = fake_images

    def on_predict_end(self):
        self.logger.experiment.add_images("Predict/img", Unormalize(self.imgs[:self.args.show_num]), self.current_epoch)
        self.logger.experiment.add_images("Predict/source_image", Unormalize(self.source_imgs[:self.args.show_num]), self.current_epoch)
        self.logger.experiment.add_images("Predict/fake_image", Unormalize(self.fake_images[:self.args.show_num]), self.current_epoch)

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