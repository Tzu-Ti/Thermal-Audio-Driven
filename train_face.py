__author__ = 'Titi'
from multiprocessing import cpu_count
from typing import Any, Optional
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.utilities.types import STEP_OUTPUT
from pytorch_lightning.strategies import DDPStrategy

from torch.utils.data import DataLoader
import torch.optim as optim
from torch import nn
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torchvision
import einops

import argparse
import os
import numpy as np

from data.dataset import AudioDataset, Unormalize
from models.wav2lip import Wav2Lip, Wav2Lip_disc_qual
from models.syncnet import SyncNet_color as SyncNet
from face import YOLOv8_face

from criterion import VGGLoss, CosineLoss

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # training setting
    parser.add_argument('--model_name', default='test')
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-4)
    # Model setting
    parser.add_argument('--filter', type=int, default=32)
    parser.add_argument('--z_dim', type=int, default=256)
    # Mode
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--predict', action="store_true")
    parser.add_argument('--ckpt_path')
    # I/O
    parser.add_argument('--root', default='/root/LRS2/mvlrs_v1/main')
    parser.add_argument('--train_txt', default='/root/LRS2/trainlst.txt')
    parser.add_argument('--val_txt', default='/root/LRS2/vallst.txt')
    parser.add_argument('--test_txt', default='/root/LRS2/testlst.txt')
    parser.add_argument('--pred_txt', default='../thermal_data/predList.txt')
    parser.add_argument('--sync_weight_path', default='weights/lipsync_expert.pth')
    # loss weight
    parser.add_argument('--w_sync', type=float, default=0.0)
    # choose
    parser.add_argument('--no_ganFeat_loss', action="store_true")
    parser.add_argument('--no_vgg_loss', action="store_true")
    parser.add_argument('--l1_loss', action="store_true")
    # visualization
    parser.add_argument('--show_num', type=int, default=4)
    # face
    parser.add_argument('--face_model_path', default='weights/yolov8n-face.onnx')
    parser.add_argument('--seq_length', type=int, default=6)
    parser.add_argument('--face_size', type=int, default=96)
    parser.add_argument('--size', type=int, default=128)
    parser.add_argument('--expansion', type=int, default=16)

    return parser.parse_args()

class Model_factory(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.automatic_optimization = False

        self.args = args
        print(">> Initialize model...")
        self.netG = Wav2Lip()
        # self.netD = Wav2Lip_disc_qual()
        self.netSyncD = SyncNet()
        netSyncD_weight = torch.load(args.sync_weight_path)
        self.netSyncD.load_state_dict(netSyncD_weight['state_dict'])

        self.criterionCosine = CosineLoss()
        self.criterionL1 = nn.L1Loss()
        self.criterionL2 = nn.MSELoss()
        self.criterionBCE = nn.BCELoss()
        self.criterionVGG = VGGLoss()

    def get_sync_loss(self, mel_forSync, frames):
        B, T, C, H, W = frames.shape
        frames_forSync = frames[:, :, :, H//2:, :]
        frames_forSync = einops.rearrange(frames_forSync, 'B T C h W -> B (T C) h W')
        a, v = self.netSyncD(mel_forSync, frames_forSync)
        y = torch.ones(B, 1).type_as(a)
        return self.criterionCosine(a, v, y)
    
    def trainG(self, mel, input, gt, mel_forSync):
        G_losses = {}

        outputs = self.netG(mel, input)
        self.outputs = outputs
        G_losses['Sync'] = self.get_sync_loss(mel_forSync, outputs) * self.args.w_sync
        G_losses['Rec'] = self.criterionL1(outputs, gt)

        g_loss = sum(G_losses.values())

        self.g_opt.zero_grad()
        self.manual_backward(g_loss)
        self.g_opt.step()

        return G_losses

    def trainD(self, mel, input, gt, mel_forSync, YT_faces):
        D_losses = {}

        with torch.no_grad():
            self.netG.eval()
            outputs = self.netG(mel, input)
        
        real = self.netD(gt)
        real_loss = self.criterionBCE(real, torch.ones_like(real))
        fake = self.netD(outputs.detach())
        fake_loss = self.criterionBCE(fake, torch.zeros_like(fake))
        D_losses['Adv'] = real_loss + fake_loss
        
        d_loss = sum(D_losses.values())
        self.d_opt.zero_grad()
        self.manual_backward(d_loss)
        self.d_opt.step()

        return D_losses

    def training_step(self, batch, batch_idx):
        self.g_opt = self.optimizers()
        if self.current_epoch == 5: self.args.w_sync = 0.01

        input, gt, mel, mel_forSync = batch

        # train Generator
        G_losses = self.trainG(mel, input, gt, mel_forSync)
        self.log('GLoss', sum(G_losses.values()).mean(), on_step=True, prog_bar=True, logger=True)
        self.logger.experiment.add_scalars("G_loss", G_losses, self.global_step)

        self.logger.experiment.add_images("Train/Input", Unormalize(input[0, :, :3]), batch_idx)
        self.logger.experiment.add_images("Train/Wrong_img", Unormalize(input[0, :, 3:]), batch_idx)
        self.logger.experiment.add_images("Train/Outputs", Unormalize(self.outputs[0, :]), batch_idx)
        self.logger.experiment.add_images("Train/GT", Unormalize(gt[0]), batch_idx)
        
    def validation_step(self, batch, batch_idx):
        input, gt, mel, mel_forSync = batch
        outputs = self.netG(mel, input)
        self.outputs = outputs

        self.input = input
        self.gt = gt

    def on_validation_epoch_end(self):
        self.logger.experiment.add_images("Val/Wrong_img", Unormalize(self.input[0, :, 3:]), self.current_epoch)
        self.logger.experiment.add_images("Val/Outputs", Unormalize(self.outputs[0, :]), self.current_epoch)
        self.logger.experiment.add_images("Val/GT", Unormalize(self.gt[0]), self.current_epoch)

    def test_step(self, batch, batch_idx):
        input, gt, mel, mel_forSync, frames_forSync, y_forSync = batch

        frames_forSync = einops.rearrange(frames_forSync, 'B T C H W -> B (T C) H W')
        audio_embedding, face_embedding = self.netSyncD(mel_forSync, frames_forSync)

        CosineSimilarity = self.criterionCosine(audio_embedding, face_embedding, y_forSync)

        print(y_forSync)
        print(CosineSimilarity)

    # def on_test_end(self):
    #     self.logger.experiment.add_images("Test/img", Unormalize(self.img[:self.args.show_num]), self.current_epoch)
    #     self.logger.experiment.add_images("Test/source_image", Unormalize(self.source_img[:self.args.show_num]), self.current_epoch)
    #     self.logger.experiment.add_images("Test/fake_image", Unormalize(self.fake_image[:self.args.show_num]), self.current_epoch)

    def configure_optimizers(self):
        G_params = list(self.netG.parameters())
        # D_params = list(self.netD.parameters())
        G_opt = optim.AdamW(G_params, lr=self.args.lr)
        # D_opt = optim.AdamW(D_params, lr=self.args.lr)
        return G_opt

    
    


    
def main():
    args = parse()

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.getcwd(),
        version=args.model_name,
        name='lightning_logs'
    )

    trainDataset = AudioDataset(root=args.root, txt_path=args.train_txt, train=True)
    trainDataloader = DataLoader(dataset=trainDataset,
                                 batch_size=args.batch_size,
                                 shuffle=True, 
                                 num_workers=cpu_count())
    valDataset = AudioDataset(root=args.root, txt_path=args.val_txt, train=False)
    valDataloader = DataLoader(dataset=valDataset,
                               batch_size=args.batch_size,
                               shuffle=False, 
                               num_workers=cpu_count())

    Model = Model_factory(args)

    trainer = pl.Trainer(fast_dev_run=True, logger=tb_logger, accelerator='gpu', devices=[0])
    trainer = pl.Trainer(max_epochs=args.epochs, check_val_every_n_epoch=5,
                         logger=tb_logger, log_every_n_steps=5,
                         strategy=DDPStrategy(find_unused_parameters=True))

    if args.train:
        trainer.fit(model=Model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader)
    elif args.resume:
        trainer.fit(model=Model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader, ckpt_path=args.ckpt_path)
    elif args.test:
        testDataset = AudioDataset(root=args.root, txt_path=args.val_txt, train=False)
        testDataloader = DataLoader(dataset=testDataset, batch_size=1, shuffle=False, num_workers=cpu_count())
        test_trainer = pl.Trainer(devices=1, logger=tb_logger)
        test_trainer.test(model=Model, dataloaders=testDataloader, ckpt_path=args.ckpt_path)
    elif args.predict:
        predictDataset = TherSeqDataset(root=args.root, txt_path=args.pred_txt, no_pair=True, train=False, seq_length=6)
        predDataloader = DataLoader(dataset=predictDataset, batch_size=1, shuffle=False, num_workers=cpu_count())
        predict_trainer = pl.Trainer(devices=1, logger=tb_logger)
        predict_trainer.predict(model=Model, dataloaders=predDataloader, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()