import torch
import torch.nn as nn
import torch.nn.functional as F

from models.utils import get_nonspade_norm_layer

class MultiscaleDiscriminator(nn.Module):
    def __init__(self, num_D=2, filter=32, input_nc=3+1):
        super().__init__()

        for i in range(num_D):
            subnetD = NLayerDiscriminator(filter=filter, input_nc=input_nc)
            self.add_module('discriminator_{}'.format(i), subnetD)

    def downsample(self, input):
        return F.avg_pool2d(input, kernel_size=3, stride=2, padding=[1, 1], count_include_pad=False)
    
    def forward(self, input):
        results = []
        for name, D in self.named_children():
            out = D(input)
            results.append(out)
            input = self.downsample(input)

        return results

class NLayerDiscriminator(nn.Module):
    def __init__(self, n_layers_D=4, filter=32, input_nc=3+1):
        super().__init__()

        norm_layer = get_nonspade_norm_layer(norm_type='spectralinstance')
        sequence = [[nn.Conv2d(input_nc, filter, kernel_size=4, stride=2, padding=2),
                     nn.LeakyReLU(0.2, False)]]
        
        for n in range(1, n_layers_D):
            filter_prev = filter
            filter = min(filter * 2, 512)
            stride = 1 if n == n_layers_D-1 else 2
            sequence += [[norm_layer(nn.Conv2d(filter_prev, filter, kernel_size=4, stride=stride, padding=2)),
                          nn.LeakyReLU(0.2, False)]]
            
        sequence += [[nn.Conv2d(filter, 1, kernel_size=4, stride=1, padding=2)]]

        for n in range(len(sequence)):
            self.add_module('model{}'.format(n), nn.Sequential(*sequence[n]))

    def forward(self, input):
        results = [input]
        for submodel in self.children():
            intermediate_output = submodel(results[-1])
            results.append(intermediate_output)

        return results[1:]
    
if __name__ == '__main__':
    D = MultiscaleDiscriminator()
    input = torch.randn([4, 4, 384, 512])
    outputs = D(input)