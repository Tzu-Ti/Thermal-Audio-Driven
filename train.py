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
from face import YOLOv8_face

def parse():
    # Argument Parser
    parser = argparse.ArgumentParser()
    # training setting
    parser.add_argument('--model_name', default='test')
    parser.add_argument('--epochs', type=int, default=500)
    parser.add_argument('--batch_size', type=int, default=12)
    parser.add_argument('--lr', type=int, default=2e-4)
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
    parser.add_argument('--root', default='../thermal_data')
    parser.add_argument('--train_txt', default='../thermal_data/trainList.txt')
    parser.add_argument('--val_txt', default='../thermal_data/valList.txt')
    parser.add_argument('--test_txt', default='../thermal_data/testList.txt')
    parser.add_argument('--pred_txt', default='../thermal_data/predList.txt')
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
    # face
    parser.add_argument('--face_model_path', default='weights/yolov8n-face.onnx')
    parser.add_argument('--seq_length', type=int, default=6)

    return parser.parse_args()

class Model_factory(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.automatic_optimization = False

        self.args = args
        self.netE = ConvEncoder(filter=args.filter, z_dim=args.z_dim)
        self.netG = Generator(filter=args.filter, z_dim=args.z_dim)
        self.netD = MultiscaleDiscriminator(filter=args.filter)

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
    
    def discriminate(self, source_img, real_img, fake_img):
        _, _, H, W = real_img.shape
        source_img = F.interpolate(source_img, size=[H, W], mode='nearest')
        fake_concat = torch.cat([source_img, fake_img], dim=1)
        real_concat = torch.cat([source_img, real_img], dim=1)

        fake_and_real = torch.cat([fake_concat, real_concat], dim=0)

        discriminator_out = self.netD(fake_and_real)
        pred_fake, pred_real = self.divide_pred(discriminator_out)

        return pred_fake, pred_real
    
    def trainG(self, img, source_img):
        G_losses = {}
        fake_image, KLDLoss = self.generate(img, source_img)
        self.fake_image = fake_image
        G_losses['KLD'] = KLDLoss

        pred_fake, pred_real = self.discriminate(source_img, img, fake_image)
        G_losses['GAN'] = self.criterionGAN(pred_fake, True, for_discriminator=False)

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

    def training_step(self, batch, batch_idx):
        self.g_opt, self.d_opt = self.optimizers()

        img, source_img = batch

        # train Generator
        G_losses = self.trainG(img, source_img)

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
        img, source_img = batch
        fake_image, _ = self.generate(img, source_img)

        self.img = img
        self.source_img = source_img
        self.fake_image = fake_image

    def on_validation_epoch_end(self):
        self.logger.experiment.add_images("Val/img", Unormalize(self.img[:self.args.show_num]), self.current_epoch)
        self.logger.experiment.add_images("Val/source_image", Unormalize(self.source_img[:self.args.show_num]), self.current_epoch)
        self.logger.experiment.add_images("Val/fake_image", Unormalize(self.fake_image[:self.args.show_num]), self.current_epoch)

    def test_step(self, batch, batch_idx):
        img, source_img = batch
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
        G_opt = optim.AdamW(G_params, lr=self.args.lr)
        D_opt = optim.AdamW(D_params, lr=self.args.lr)
        return [G_opt, D_opt]
    
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
    
    def predict_step(self, batch, batch_idx):
        print("Initial YOLOv8 face detector")
        YOLOv8_face_detector = YOLOv8_face(self.args.face_model_path, conf_thres=0.45, iou_thres=0.5)
        face_transforms = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize([128, 128])
        ])

        img, source_imgs = batch
        source_imgs = torch.cat(source_imgs, dim=0)
        imgs = torch.cat([img for i in range(self.args.seq_length)], dim=0)
        
        fake_images, _ = self.generate(imgs, source_imgs)

        coord, ref_face = detectFace(detector=YOLOv8_face_detector, img=img[0])
        ref_face = face_transforms(ref_face)
        
        fake_faces = []
        new_fake_images = []
        for fake in fake_images:
            coord, fake_face = detectFace(detector=YOLOv8_face_detector, img=fake)
            x1, y1, x2, y2 = coord
            w = abs(x1 - x2)
            h = abs(y1 - y2)
            new_ref_face = TF.resize(ref_face, [h, w])
            new_fake_image = fake.clone()
            new_fake_image[:, y1:y2, x1:x2] = new_ref_face
            new_fake_image = new_fake_image.unsqueeze(0)
            new_fake_images.append(new_fake_image)

            fake_face = face_transforms(fake_face)
            fake_face = fake_face.unsqueeze(0)
            fake_faces.append(fake_face)

        self.imgs = imgs
        self.source_imgs = source_imgs
        self.fake_images = fake_images
        self.ref_face = ref_face
        self.fake_faces = torch.cat(fake_faces, dim=0)
        self.new_fake_images = torch.cat(new_fake_images, dim=0)

    def on_predict_end(self):
        self.logger.experiment.add_images("Predict/img", Unormalize(self.imgs[:self.args.show_num]), self.current_epoch)
        self.logger.experiment.add_images("Predict/source_image", Unormalize(self.source_imgs[:self.args.show_num]), self.current_epoch)
        self.logger.experiment.add_images("Predict/fake_image", Unormalize(self.fake_images[:self.args.show_num]), self.current_epoch)
        self.logger.experiment.add_images("Predict/new_fake_image", Unormalize(self.new_fake_images[:self.args.show_num]), self.current_epoch)
        self.logger.experiment.add_image("Predict/ref_face", Unormalize(self.ref_face), self.current_epoch)
        self.logger.experiment.add_images("Predict/fake_face", Unormalize(self.fake_faces[:self.args.show_num]), self.current_epoch)

    
def detectFace(detector, img):
    img = img.cpu().detach().numpy().transpose(1, 2, 0)

    boxes, scores, classids, kpts = detector.detect(img)
    x, y, w, h = boxes[0].astype(int)
    bigger = 10
    if w >= h: 
        new_h = w
        y_complement = (new_h - h) // 2 + bigger
        x_complement = 0 + bigger
    elif h > w:
        new_w = h
        x_complement = (new_w - w) // 2 + bigger
        y_complement = 0 + bigger
    y1 = y - y_complement
    y2 = y + h + y_complement
    x1 = x - x_complement
    x2 = x + w + x_complement
    face = img[y1: y2, x1: x2, :]
    return [x1, y1, x2, y2], face
    
    


    
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

    Model = Model_factory(args)

    # trainer = pl.Trainer(fast_dev_run=True, logger=tb_logger, accelerator='gpu', devices=[0, 1])
    trainer = pl.Trainer(max_epochs=args.epochs, check_val_every_n_epoch=5,
                         logger=tb_logger, log_every_n_steps=5,
                         strategy=DDPStrategy(find_unused_parameters=True))

    if args.train:
        trainer.fit(model=Model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader)
    elif args.resume:
        trainer.fit(model=Model, train_dataloaders=trainDataloader, val_dataloaders=valDataloader, ckpt_path=args.ckpt_path)
    elif args.test:
        test_trainer = pl.Trainer(devices=1, logger=tb_logger)
        test_trainer.test(model=Model, dataloaders=testDataloader, ckpt_path=args.ckpt_path)
    elif args.predict:
        predictDataset = TherSeqDataset(root=args.root, txt_path=args.pred_txt, no_pair=True, train=False, seq_length=6)
        predDataloader = DataLoader(dataset=predictDataset, batch_size=1, shuffle=False, num_workers=cpu_count())
        predict_trainer = pl.Trainer(devices=1, logger=tb_logger)
        predict_trainer.predict(model=Model, dataloaders=predDataloader, ckpt_path=args.ckpt_path)

if __name__ == '__main__':
    main()