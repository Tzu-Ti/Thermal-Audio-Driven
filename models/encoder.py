import torch
import torch.nn as nn
from .utils import get_nonspade_norm_layer

class ConvEncoder(nn.Module):
    """ Same architecture as the image discriminator """
    def __init__(self, filter=32, z_dim=256):
        super().__init__()

        norm_layer = get_nonspade_norm_layer(norm_type='spectralinstance')
        self.layer1 = norm_layer(nn.Conv2d(3, filter, 3, stride=2, padding=1)) # 192, 256
        self.layer2 = norm_layer(nn.Conv2d(filter * 1, filter * 2, 3, stride=2, padding=1)) #  96, 128
        self.layer3 = norm_layer(nn.Conv2d(filter * 2, filter * 4, 3, stride=2, padding=1)) #  48,  64
        self.layer4 = norm_layer(nn.Conv2d(filter * 4, filter * 8, 3, stride=2, padding=1)) #  24,  32
        self.layer5 = norm_layer(nn.Conv2d(filter * 8, filter * 8, 3, stride=2, padding=1)) #  12,  16
        self.layer6 = norm_layer(nn.Conv2d(filter * 8, filter * 8, 3, stride=2, padding=1)) #   6,   8

        self.actvn = nn.LeakyReLU(0.2, False)

        sh, sw = [6, 8]
        self.fc_mu = nn.Linear(filter * 8 * sh * sw, z_dim)
        self.fc_var = nn.Linear(filter * 8 * sh * sw, z_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar
    
class FaceEncoder(nn.Module):
    """ Same architecture as the image discriminator """
    def __init__(self, filter=64, z_dim=512):
        super().__init__()

        norm_layer = get_nonspade_norm_layer(norm_type='spectralinstance')
        self.layer1 = norm_layer(nn.Conv2d(6, filter, 3, stride=2, padding=1)) # 48, 48
        self.layer2 = norm_layer(nn.Conv2d(filter * 1, filter * 2, 3, stride=2, padding=1)) #  24, 24
        self.layer3 = norm_layer(nn.Conv2d(filter * 2, filter * 4, 3, stride=2, padding=1)) #  12, 12
        self.layer4 = norm_layer(nn.Conv2d(filter * 4, filter * 8, 3, stride=2, padding=1)) #  6, 6
        self.layer5 = norm_layer(nn.Conv2d(filter * 8, filter * 8, 3, stride=2, padding=1)) #  3, 3
        self.layer6 = nn.Conv2d(filter * 8, filter * 8, 3, stride=1, padding=0) #  1, 1
        self.actvn = nn.LeakyReLU(0.2, False)

        self.fc = nn.Linear(filter*8, z_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        x = self.layer6(self.actvn(x))
        x = self.actvn(x)

        x = x.view(x.size(0), -1)
        z = self.fc(x)

        return z

if __name__ == '__main__':
    # input = torch.randn([4, 3, 384, 512])
    # E = ConvEncoder()
    # mu, logvar = E(input)
    # print(mu.shape, logvar.shape)
    input = torch.randn([4, 3, 96, 96])
    E = FaceEncoder()
    f = E(input)