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

from data.dataset import TherDataset, TherSeqDataset, Unormalize
from models.encoder import ConvEncoder
from models.generator import Generator
from models.discriminator import MultiscaleDiscriminator
from criterion import KLDLoss, GANLoss, VGGLoss

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # training setting
    parser.add_argument('--model_name', default='test')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=int, default=2e-4)
    # Model setting
    parser.add_argument('--filter', type=int, default=32)
    parser.add_argument('--z_dim', type=int, default=256)
    # Mode
    parser.add_argument('--with_pretrainIDD', action="store_true")
    parser.add_argument('--pretrainIDD', action="store_true")
    parser.add_argument('--train', action="store_true")
    parser.add_argument('--test', action="store_true")
    parser.add_argument('--resume', action="store_true")
    parser.add_argument('--ckpt_path')
    parser.add_argument('--pretrainIDD_ckpt_path')
    # I/O
    parser.add_argument('--root', default='/root/thermal_data')
    parser.add_argument('--train_txt', default='/root/thermal_data/trainList.txt')
    parser.add_argument('--val_txt', default='/root/thermal_data/valList.txt')
    parser.add_argument('--test_txt', default='/root/thermal_data/testList.txt')
    # loss weight
    parser.add_argument('--w_kld', type=float, default=0.05)
    parser.add_argument('--w_feat', type=float, default=10.0)
    parser.add_argument('--w_vgg', type=float, default=10.0)
    parser.add_argument('--w_rec', type=float, default=10.0)
    # choose
    parser.add_argument('--no_ganFeat_loss', action="store_true")
    parser.add_argument('--no_vgg_loss', action="store_true")
    parser.add_argument('--l1_loss', action="store_true")
    # visualization
    parser.add_argument('--show_num', type=int, default=4)

    return parser.parse_args()

class Model_factory(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.automatic_optimization = False

        self.args = args
        self.netE = ConvEncoder(filter=args.filter, z_dim=args.z_dim)
        self.netG = Generator(filter=args.filter, z_dim=args.z_dim)
        self.netD = MultiscaleDiscriminator(filter=args.filter)
        self.netIDD = MultiscaleDiscriminator(num_D=1, filter=args.filter, input_nc=6)

        self.KLDLoss = KLDLoss()
        self.criterionGAN = GANLoss()
        self.criterionFeat = nn.L1Loss()
        self.criterionVGG = VGGLoss()
        self.criterionRec = nn.L1Loss()

    def generate(self, img, source_img):
        mu, logvar = self.netE(img)
        z = self.reparameterize(mu, logvar)

        fake_image = self.netG(source_img, z)

        KLDLoss = self.KLDLoss(mu, logvar) * self.args.w_kld
        return fake_image, KLDLoss
    
    def trainG(self, img, source_img, wrong_img):
        G_losses = {}
        fake_image, KLDLoss = self.generate(img, source_img)
        self.fake_image = fake_image
        G_losses['KLD'] = KLDLoss

        pred_fake, pred_real = self.discriminate(source_img, img, fake_image)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

        id_pred_fake, id_pred_real = self.discriminateIDD(fake_image, img, wrong_img)
        G_losses['IDGAN'] = self.criterionGAN(id_pred_fake, True, for_discriminator=False)

        if not self.args.no_ganFeat_loss:
            num_D = len(pred_fake)
            GAN_Feat_loss = torch.cuda.FloatTensor(1).fill_(0)
            for i in range(num_D):
                # last output is the final prediction, so we exclude it
                num_intermediate_outputs = len(pred_fake[i]) - 1
                for j in range(num_intermediate_outputs):
                    unweighted_loss = self.criterionFeat(pred_fake[i][j], pred_real[i][j].detach())
                    GAN_Feat_loss += unweighted_loss * self.args.w_feat
            G_losses['GAN_Feat'] = GAN_Feat_loss

        if not self.args.no_vgg_loss:
            G_losses['VGG'] = self.criterionVGG(fake_image, img) * self.args.w_vgg

        if self.args.l1_loss:
            G_losses['Rec'] = self.criterionRec(fake_image, img) * self.args.w_rec

        g_loss = sum(G_losses.values()).mean()
        self.g_opt.zero_grad()
        self.manual_backward(g_loss)
        self.g_opt.step()

        return G_losses

    def discriminate(self, source_img, real_img, fake_img):
        _, _, H, W = real_img.shape
        source_img = F.interpolate(source_img, size=[H, W], mode='nearest')
        fake_concat = torch.cat([source_img, fake_img], dim=1)
        real_concat = torch.cat([source_img, real_img], dim=1)

        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real

    def trainD(self, img, source_img):
        D_losses = {}
        with torch.no_grad():
            fake_image, _ = self.generate(img, source_img)
            fake_image = fake_image.detach()
            fake_image.requires_grad_()
        
        pred_fake, pred_real = self.discriminate(source_img, img, fake_image)

        D_losses['D_fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        D_losses['D_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)

        d_loss = sum(D_losses.values()).mean()
        self.d_opt.zero_grad()
        self.manual_backward(d_loss)
        self.d_opt.step()

        return D_losses
    
    def discriminateIDD(self, img, pair_img, wrong_img):
        pair_inputs = torch.cat([img, pair_img], dim=1)
        wrong_inputs = torch.cat([img, wrong_img], dim=1)
        inputs = torch.cat([wrong_inputs, pair_inputs], dim=0)

        discriminator_out = self.netIDD(inputs)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real
    
    def trainIDD(self, img, pair_img, wrong_img):
        IDD_losses = {}

        pred_fake, pred_real = self.discriminateIDD(img, pair_img, wrong_img)

        IDD_losses['IDD_fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
        IDD_losses['IDD_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)

        idd_loss = sum(IDD_losses.values()).mean()
        self.idd_opt.zero_grad()
        self.manual_backward(idd_loss)
        self.idd_opt.step()

        return IDD_losses
        
    def training_step(self, batch, batch_idx):
        self.g_opt, self.d_opt, self.idd_opt = self.optimizers()

        img, source_img, pair_img, wrong_img = batch

        if self.args.pretrainIDD:
            IDD_losses = self.trainIDD(img, pair_img, wrong_img)
            
            self.log_dict({
                'IDDLoss': sum(IDD_losses.values()).mean()
            }, on_step=True, prog_bar=True, logger=True, sync_dist=True)
            self.logger.experiment.add_images("Train/img", Unormalize(img[:self.args.show_num]), batch_idx)
            self.logger.experiment.add_images("Train/wrong_image", Unormalize(wrong_img[:self.args.show_num]), batch_idx)

        else:
            # train Generator
            G_losses = self.trainG(img, source_img, wrong_img)

            # train Discriminator
            D_losses = self.trainD(img, source_img)

            self.log_dict({
                'GLoss': sum(G_losses.values()).mean(),
                'DLoss': sum(D_losses.values()).mean()
            }, on_step=True, prog_bar=True, logger=True, sync_dist=True)
            self.logger.experiment.add_images("Train/img", Unormalize(img[:self.args.show_num]), batch_idx)
            self.logger.experiment.add_images("Train/source_image", Unormalize(source_img[:self.args.show_num]), batch_idx)
            self.logger.experiment.add_images("Train/fake_image", Unormalize(self.fake_image[:self.args.show_num]), batch_idx)

    def validation_step(self, batch, batch_idx):
        img, source_img, pair_img, wrong_img = batch

        if self.args.pretrainIDD:
            IDD_losses = {}
            pred_fake, pred_real = self.discriminateIDD(img, pair_img, wrong_img)

            IDD_losses['IDD_fake'] = self.criterionGAN(pred_fake, False, for_discriminator=True)
            IDD_losses['IDD_real'] = self.criterionGAN(pred_real, True, for_discriminator=True)

            self.log_dict(IDD_losses, on_step=True, prog_bar=True, logger=True, sync_dist=True)

        else:
            fake_image, _ = self.generate(img, source_img)

            self.img = img
            self.source_img = source_img
            self.fake_image = fake_image

    def on_validation_epoch_end(self):
        if self.args.pretrainIDD:
            pass
        else:
            self.logger.experiment.add_images("Val/img", Unormalize(self.img[:self.args.show_num]), self.current_epoch)
            self.logger.experiment.add_images("Val/source_image", Unormalize(self.source_img[:self.args.show_num]), self.current_epoch)
            self.logger.experiment.add_images("Val/fake_image", Unormalize(self.fake_image[:self.args.show_num]), self.current_epoch)

    def test_step(self, batch, batch_idx):
        img, source_img, pair_img, wrong_img = batch
        fake_image, _ = self.generate(img, source_img)

        self.img = img
        self.source_img = source_img
        self.fake_image = fake_image

    def on_test_end(self):
        self.logger.experiment.add_images("Test/img", Unormalize(self.img[:self.args.show_num]), self.current_epoch)
        self.logger.experiment.add_images("Test/source_image", Unormalize(self.source_img[:self.args.show_num]), self.current_epoch)
        self.logger.experiment.add_images("Test/fake_image", Unormalize(self.fake_image[:self.args.show_num]), self.current_epoch)

    def configure_optimizers(self):
        G_params = list(self.netG.parameters()) + list(self.netE.parameters())
        D_params = list(self.netD.parameters())
        IDD_params = list(self.netIDD.parameters())
        G_opt = optim.AdamW(G_params, lr=self.args.lr)
        D_opt = optim.AdamW(D_params, lr=self.args.lr)
        IDD_opt = optim.AdamW(IDD_params, lr=self.args.lr)
        return [G_opt, D_opt, IDD_opt]
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps.mul(std) + mu
    
    def divide_pred(self, pred):
        if type(pred) == list:
            fake = []
            real = []
            for p in pred:
                fake.append([tensor[:tensor.size(0) // 2] for tensor in p])
                real.append([tensor[tensor.size(0) // 2:] for tensor in p])
        else:
            fake = pred[:pred.size(0) // 2]
            real = pred[pred.size(0) // 2:]

        return fake, real

    
def main():
    args = parse()

    tb_logger = pl_loggers.TensorBoardLogger(
        save_dir=os.getcwd(),
        version=args.model_name,
        name='lightning_logs'
    )

    trainDataset = TherDataset(root=args.root, txt_path=args.train_txt, train=True)
    trainDataloader = DataLoader(dataset=trainDataset,
                                 batch_size=args.batch_size,
                                 shuffle=True, 
                                 num_workers=cpu_count())
    valDataset = TherDataset(root=args.root, txt_path=args.val_txt, train=False)
    valDataloader = DataLoader(dataset=valDataset,
                               batch_size=args.batch_size,
                               shuffle=False, 
                               num_workers=cpu_count())
    testDataset = TherDataset(root=args.root, txt_path=args.val_txt, no_pair=True, train=False)
    testDataloader = DataLoader(dataset=testDataset,
                                batch_size=args.batch_size,
                                shuffle=True, 
                                num_workers=cpu_count())

    if args.with_pretrainIDD:
        Model = Model_factory.load_from_checkpoint(args.pretrainIDD_ckpt_path, args=args)
    else:
        Model = Model_factory(args)

    # trainer = pl.Trainer(fast_dev_run=True, logger=tb_logger, accelerator='gpu', devices=[0, 1])
    trainer = pl.Trainer(max_epochs=args.epochs, check_val_every_n_epoch=1,
                         logger=tb_logger, log_every_n_steps=5,
                         strategy=DDPStrategy(find_unused_parameters=True))

    if args.train:
        trainer.fit(model=Model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader)
    elif args.resume:
        trainer.fit(model=Model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader, ckpt_path=args.ckpt_path)
    elif args.test:
        test_trainer = pl.Trainer(devices=1, logger=tb_logger)
        test_trainer.test(model=Model, dataloaders=testDataloader, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()