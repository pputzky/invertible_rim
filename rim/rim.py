import torch
from torch import nn

class RIM(nn.Module):

    def __init__(self, rnn, grad_fun):
        super(RIM, self).__init__()
        self.rnn = rnn
        self.grad_fun = grad_fun

    def forward(self, eta, data, hx=None, n_steps=1, accumulate_eta=False):
        """
        :param eta: Starting value for eta [n_batch,features,height,width]
        :param grad_fun: The gradient function, takes as input eta and outputs gradient of same dimensionality
        :param hx: Hidden state of the RNN
        :param n_steps: Number of time steps, that the RIM should perform. Default: 1
        :param accumulate_eta: Bool, if True will save all intermediate etas in a list, else outputs only the last eta.
                               Default: False
        :return: etas, hx
        """
        etas = []

        for i in range(n_steps):
            grad_eta = self.grad_fun(eta, data)
            x_in = torch.cat((eta, grad_eta), 1)

            delta, hx = self.rnn.forward(x_in, hx)
            eta = eta + delta

            if accumulate_eta:
                etas.append(eta)

        if not accumulate_eta:
            etas = eta

        return etas, hx

