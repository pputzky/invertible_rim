import torch
import torch.nn as nn

class InputRNN(torch.nn.Module):

    def __init__(self, rnn_cell=None, input_fun=None):
        super(InputRNN, self).__init__()

        self.rnn_cell = rnn_cell
        self.input_fun = input_fun

    def forward(self, x, hx=None):

        if self.input_fun is not None:
            x = self.input_fun.forward(x)
        if self.rnn_cell is not None:
            x = self.rnn_cell.forward(x)
            hx = x

        return x, hx

class ConvNonlinear(nn.Module):

    def __init__(self, input_size, features, conv_dim, kernel_size, dilation, bias, nonlinear='relu'):
        super(ConvNonlinear, self).__init__()

        self.input_size = input_size
        self.features = features
        self.bias = bias
        self.conv_dim = conv_dim
        self.conv_class = self.determine_conv_class(conv_dim)
        if nonlinear is not None and nonlinear.upper() == 'RELU':
            self.nonlinear = torch.nn.ReLU()
        elif nonlinear is None:
            self.nonlinear = lambda x: x
        else:
            ValueError('Please specify a proper')

        self.padding = [torch.nn.ReplicationPad1d(dilation * (kernel_size-1) // 2),
                        torch.nn.ReplicationPad2d(dilation * (kernel_size-1) // 2),
                        torch.nn.ReplicationPad3d(dilation * (kernel_size - 1) // 2)][conv_dim-1]
        self.conv_layer =  self.conv_class(in_channels=input_size, out_channels=features,
                             kernel_size=kernel_size, padding=0,
                             dilation=dilation, bias=bias)

        self.reset_parameters()

    def reset_parameters(self):
        torch.nn.init.kaiming_normal_(self.conv_layer.weight, nonlinearity='relu')

        if self.conv_layer.bias is not None:
            nn.init.zeros_(self.conv_layer.bias)

    def determine_conv_class(self, n_dim):

        if n_dim is 1:
            return nn.Conv1d
        elif n_dim is 2:
            return nn.Conv2d
        elif n_dim is 3:
            return nn.Conv3d
        else:
            NotImplementedError("No convolution of this dimensionality implemented")

    def extra_repr(self):
        s = '{input_size}, {features}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinear' in self.__dict__ and self.nonlinear != "tanh":
            s += ', nonlinearity={nonlinear}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def forward(self, input):
        return self.nonlinear(self.conv_layer(self.padding(input)))


class ConvRNNCellBase(nn.Module):

    def __init__(self, input_size, hidden_size, num_chunks, conv_dim, kernel_size,
                dilation, bias):
        super(ConvRNNCellBase, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias
        self.num_chunks = num_chunks
        self.conv_dim = conv_dim
        self.conv_class = self.determine_conv_class(conv_dim)
        self.ih = self.conv_class(in_channels=input_size, out_channels=num_chunks*hidden_size,
                             kernel_size=kernel_size, padding=dilation * (kernel_size-1)//2,
                             dilation=dilation, bias=bias)
        self.hh = self.conv_class(in_channels=hidden_size, out_channels=num_chunks*hidden_size,
                             kernel_size=kernel_size, padding=dilation * (kernel_size-1)//2,
                             dilation=dilation, bias=bias)
        self.reset_parameters()

    def reset_parameters(self):

        self.ih.weight.data = self.orthotogonalize_weights(self.ih.weight.data)
        self.hh.weight.data = self.orthotogonalize_weights(self.hh.weight.data)

        if self.bias is True:
            nn.init.zeros_(self.ih.bias)
            nn.init.zeros_(self.hh.bias)

    def orthotogonalize_weights(self, weights, chunks=1):
        return torch.cat([nn.init.orthogonal_(w) for w in weights.chunk(chunks,0)],0)

    def determine_conv_class(self, n_dim):

        if n_dim is 1:
            return nn.Conv1d
        elif n_dim is 2:
            return nn.Conv2d
        elif n_dim is 3:
            return nn.Conv3d
        else:
            NotImplementedError("No convolution of this dimensionality implemented")

    def extra_repr(self):
        s = '{input_size}, {hidden_size}'
        if 'bias' in self.__dict__ and self.bias is not True:
            s += ', bias={bias}'
        if 'nonlinearity' in self.__dict__ and self.nonlinearity != "tanh":
            s += ', nonlinearity={nonlinearity}'
        return s.format(**self.__dict__)

    def check_forward_input(self, input):
        if input.size(1) != self.input_size:
            raise RuntimeError(
                "input has inconsistent input_size: got {}, expected {}".format(
                    input.size(1), self.input_size))

    def check_forward_hidden(self, input, hx, hidden_label=''):
        if input.size(0) != hx.size(0):
            raise RuntimeError(
                "Input batch size {} doesn't match hidden{} batch size {}".format(
                    input.size(0), hidden_label, hx.size(0)))

        if hx.size(1) != self.hidden_size:
            raise RuntimeError(
                "hidden{} has inconsistent hidden_size: got {}, expected {}".format(
                    hidden_label, hx.size(1), self.hidden_size))


class ConvGRUCell(ConvRNNCellBase):
    """
    This is an implementation of a Convolutional GRU Cell following the Pytorch implementation
    of a GRU. Here, the fully connected linear transforms are simply replaced by convolutional
    linear transforms.
    """

    def __init__(self, input_size, hidden_size, conv_dim, kernel_size, dilation=1, bias=True):
        super(ConvGRUCell, self).__init__(input_size=input_size, hidden_size=hidden_size,
                                          num_chunks=3, conv_dim=conv_dim, kernel_size=kernel_size,
                                          dilation=dilation, bias=bias)

    def forward(self, input, hx=None):
        # self.check_forward_input(input)
        if hx is None:
            hx = input.new_zeros((input.size(0), self.hidden_size) + input.size()[2:],
                                 requires_grad=False)
        # self.check_forward_hidden(input, hx)

        ih = self.ih(input).chunk(3,dim=1)
        hh = self.hh(hx).chunk(3,dim=1)

        z = torch.sigmoid(ih[0] + hh[0])
        r = torch.sigmoid(ih[1] + hh[1])
        n = torch.tanh(ih[2] + r*hh[2])

        hx = (1. - z) * hx + z * n

        return hx

class ConvRNN(nn.Module):

    def __init__(self, input_size,
                 conv_params={'features':[64, 64, 2], 'k_size':[5, 3, 3],'dilation':[1, 2, 1],'bias':[True,True,False],
                              'nonlinear':['relu','relu',None]},
                 rnn_params={'features':[64, 64, 0], 'k_size':[1, 1, 0], 'dilation':[1, 1, 0], 'bias': [True,True, False],
                             'rnn_type': ['gru','gru',None]},
                 conv_dim=2):
        '''
        Generates a multi-layer convolutional GRU.
        Preserves spatial dimensions across cells, only altering depth.

        Parameters
        ----------
        input_size : integer. depth dimension of input tensors.
        hidden_sizes : integer or list. depth dimensions of hidden state.
            if integer, the same hidden size is used for all cells.
        kernel_sizes : integer or list. sizes of Conv2d gate kernels.
            if integer, the same kernel size is used for all cells.
        n_layers : integer. number of chained `ConvGRUCell`.
        '''

        super(ConvRNN, self).__init__()

        self.input_size = input_size
        self.conv_dim = conv_dim
        self.conv_params = conv_params
        self.rnn_params = rnn_params

        conv_params = zip(*(conv_params[k] for k in ['features', 'k_size', 'dilation', 'bias', 'nonlinear']))
        rnn_params = zip(*(rnn_params[k] for k in ['features', 'k_size', 'dilation', 'bias', 'rnn_type']))

        self.layers = nn.ModuleList()
        for (conv_features, conv_k_size, conv_dilation, conv_bias, nonlinear), \
            (rnn_features, rnn_k_size, rnn_dilation, rnn_bias, rnn_type) in zip(conv_params, rnn_params):
            conv_layer = None
            rnn_layer = None

            if conv_features > 0:
                conv_layer = ConvNonlinear(input_size, conv_features, conv_dim=self.conv_dim,
                                           kernel_size=conv_k_size, dilation=conv_dilation, bias=conv_bias,
                                           nonlinear=nonlinear)
                input_size = conv_features

            if rnn_features > 0 and rnn_type is not None:
                if rnn_type.upper() == 'GRU':
                    rnn_type = ConvGRUCell
                elif issubclass(rnn_type, ConvRNNCellBase):
                    rnn_type = rnn_type
                else:
                    ValueError('Please speacify a proper rrn_type')

                rnn_layer = rnn_type(input_size, rnn_features, conv_dim=self.conv_dim,
                                          kernel_size=rnn_k_size, dilation=rnn_dilation, bias=rnn_bias)
                input_size = rnn_features

            self.layers.append(InputRNN(rnn_layer, conv_layer))

    def forward(self, input, hx=None):
        if not hx:
            hx = [None]*len(self.layers)

        hidden_new = []

        for layer, local_hx in zip(self.layers,hx):
            input, new_hx = layer.forward(input, local_hx)
            hidden_new.append(new_hx)

        return input, hidden_new
