import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import SPADEResnetBlock

class Generator(nn.Module):
    def __init__(self, filter=32, z_dim=256):
        super().__init__()
        self.filter = filter

        self.sh, self.sw = sh, sw = [6, 8]
        self.fc = nn.Linear(z_dim, 16 * filter * sh * sw)

        self.head_0 = SPADEResnetBlock(fin=16 * filter, fout=16 * filter)

        self.G_middle_0 = SPADEResnetBlock(fin=16 * filter, fout=16 * filter)
        self.G_middle_1 = SPADEResnetBlock(fin=16 * filter, fout=16 * filter)

        self.up_0 = SPADEResnetBlock(fin=16 * filter, fout=8 * filter)
        self.up_1 = SPADEResnetBlock(fin= 8 * filter, fout=4 * filter)
        self.up_2 = SPADEResnetBlock(fin= 4 * filter, fout=2 * filter)
        self.up_3 = SPADEResnetBlock(fin= 2 * filter, fout=1 * filter)

        self.conv_img = nn.Conv2d(filter, 3, kernel_size=3, padding=1)

        self.up = nn.Upsample(scale_factor=2)

    def forward(self, source_img, z):
        x = self.fc(z)
        x = x.view(-1, 16 * self.filter, self.sh, self.sw)

        x = self.head_0(x, source_img) # 6, 8

        x = self.up(x) # 12, 16
        x = self.G_middle_0(x, source_img)
        x = self.up(x) # 24, 32
        x = self.G_middle_1(x, source_img)

        x = self.up(x) #  48,  64
        x = self.up_0(x, source_img)
        x = self.up(x) #  96, 128
        x = self.up_1(x, source_img)
        x = self.up(x) # 192, 256
        x = self.up_2(x, source_img)
        x = self.up(x) # 384, 512
        x = self.up_3(x, source_img)

        x = self.conv_img(F.leaky_relu(x, 2e-1))
        x = F.sigmoid(x)

        return x


if __name__ == '__main__':
    G = Generator()
    input = torch.randn([4, 1, 12, 16])
    z = torch.randn([4, 256])
    G(input, z)