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

import argparse
import os
import numpy as np

from data.dataset import BorderDataset, Unormalize
from models.encoder import FaceEncoder
from models.generator import FaceGenerator
from criterion import VGGLoss

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # training setting
    parser.add_argument('--model_name', default='test')
    parser.add_argument('--epochs', type=int, default=3000)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=int, default=2e-4)
    # Model setting
    parser.add_argument('--border', type=int, default=8)
    # Mode
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--resume', action="store_true")
    # visualization
    parser.add_argument('--show_num', type=int, default=4)

    return parser.parse_args()

class Model_factory(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.automatic_optimization = False

        self.args = args
        self.netE = FaceEncoder()
        self.netG = FaceGenerator()

        self.criterionRec = nn.L1Loss()
        self.criterionVGG = VGGLoss()

    def generate(self, inputs):
        z = self.netE(inputs)
        fake = self.netG(z)

        return fake
    
    def get_inside_loss(self, fake, target):
        # down sample to 64x64 to calculate coarse loss
        fake = F.interpolate(fake.clone(), size=(64, 64))
        target = F.interpolate(target.clone(), size=(64, 64))
        fake = fake[:, :, self.args.border: -self.args.border, self.args.border: -self.args.border]
        target = target[:, :, self.args.border: -self.args.border, self.args.border: -self.args.border]
        
        return self.criterionRec(fake, target), self.criterionVGG(fake, target)
        
    def get_border_loss(self, fake, target):
        fake = fake.clone()
        target = target.clone()
        fake[:, :, self.args.border: -self.args.border, self.args.border: -self.args.border] = 0
        target[:, :, self.args.border: -self.args.border, self.args.border: -self.args.border] = 0

        return self.criterionRec(fake, target), self.criterionVGG(fake, target)

    def train_G(self, ref, target):
        G_losses = {}

        inputs = torch.cat([ref, target], dim=1)
        fake_image = self.generate(inputs)
        self.fake_image = fake_image

        # inside loss, inside like reference image
        G_losses['IN_L1'], G_losses['IN_VGG'] = self.get_inside_loss(fake_image, ref)
        # border loss, border like target image
        G_losses['BD_L1'], G_losses['BD_VGG'] = self.get_border_loss(fake_image, target)

        g_loss = sum(G_losses.values()).mean()
        self.g_opt.zero_grad()
        self.manual_backward(g_loss)
        self.g_opt.step()

        return G_losses

    def training_step(self, batch, batch_idx):
        self.g_opt = self.optimizers()
        ref, target = batch

        # train Generator
        G_losses = self.train_G(ref, target)

        self.log_dict({
            'GLoss': sum(G_losses.values()).mean(),
        }, on_step=True, prog_bar=True, logger=True, sync_dist=True)
        self.logger.experiment.add_images("Train/ref", Unormalize(ref[:self.args.show_num]), batch_idx)
        self.logger.experiment.add_images("Train/target", Unormalize(target[:self.args.show_num]), batch_idx)
        self.logger.experiment.add_images("Train/fake_image", Unormalize(self.fake_image[:self.args.show_num]), batch_idx)

    def configure_optimizers(self):
        G_params = list(self.netE.parameters()) + list(self.netG.parameters())
        G_opt = optim.AdamW(G_params, lr=self.args.lr)
        return G_opt

if __name__ == '__main__':
    args = parse()

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.getcwd(),
        version=args.model_name,
        name='lightning_logs'
    )

    # Dataset
    trainDataset = BorderDataset(root="/root/YouTubeFaces/frame_images_DB", txt_path="/root/YouTubeFaces/trainList.txt", return_img=False, face_size=96)
    trainDataloader = DataLoader(dataset=trainDataset,
                                 batch_size=args.batch_size,
                                 shuffle=True, 
                                 num_workers=cpu_count())
    valDataset = BorderDataset(root="/root/YouTubeFaces/frame_images_DB", txt_path="/root/YouTubeFaces/trainList.txt", return_img=True, face_size=96)
    valDataloader = DataLoader(dataset=trainDataset,
                               batch_size=1,
                               shuffle=True, 
                               num_workers=cpu_count())

    Model = Model_factory(args)

    # trainer = pl.Trainer(fast_dev_run=True, logger=tb_logger, accelerator='gpu', devices=[0, 1])
    trainer = pl.Trainer(max_epochs=args.epochs, check_val_every_n_epoch=1,
                         logger=tb_logger, log_every_n_steps=5,
                         strategy=DDPStrategy(find_unused_parameters=True))

    if args.train:
        trainer.fit(model=Model, train_dataloaders=trainDataloader) #, val_dataloaders=valDataloader)
    elif args.resume:
        trainer.fit(model=Model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader, ckpt_path=args.ckpt_path)
    elif args.test:
        test_trainer = pl.Trainer(devices=1, logger=tb_logger)
        test_trainer.test(model=Model, dataloaders=testDataloader, ckpt_path=args.ckpt_path)
