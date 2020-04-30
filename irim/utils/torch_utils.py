from torch import nn
from torch.nn import Module


def determine_conv_class(n_dim, transposed=False):
    if n_dim == 1:
        if not transposed:
            return nn.Conv1d
        else:
            return nn.ConvTranspose1d
    elif n_dim == 2:
        if not transposed:
            return nn.Conv2d
        else:
            return nn.ConvTranspose2d
    elif n_dim == 3:
        if not transposed:
            return nn.Conv3d
        else:
            return nn.ConvTranspose3d
    else:
        NotImplementedError("No convolution of this dimensionality implemented")


def determine_conv_functional(n_dim, transposed=False):
    if n_dim == 1:
        if not transposed:
            return nn.functional.conv1d
        else:
            return nn.functional.conv_transposed1d
    elif n_dim == 2:
        if not transposed:
            return nn.functional.conv2d
        else:
            return nn.functional.conv_transposed2d
    elif n_dim == 3:
        if not transposed:
            return nn.functional.conv3d
        else:
            return nn.functional.conv_transposed3d
    else:
        NotImplementedError("No convolution of this dimensionality implemented")


def pixel_unshuffle(x, downscale_factor):
    b, c, h, w = x.size()
    r = downscale_factor
    out_channel = c * (r ** 2)
    out_h = h // r
    out_w = w // r
    x_view = x.contiguous().view(b, c, out_h, r, out_w, r)
    x_prime = x_view.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, out_channel, out_h, out_w)

    return x_prime


class PixelUnshuffle(Module):
    def __init__(self, downscale_factor):
        super(PixelUnshuffle, self).__init__()
        self.downscale_factor = downscale_factor

    def forward(self, x):
        return pixel_unshuffle(x, self.downscale_factor)
