"""
An example of how to define custom invertible modules.

All Invertible Models should inherit from InvertibleModule
All Invertible Layers  should inherit from InvertibleLayer
"""
import torch

from irim import InvertibleModule, InvertibleLayer, MemoryFreeInvertibleModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ListReverse(InvertibleLayer):
    """
    A function that reverses the order of a list of Tensors. This functions implements
    the interface as defines by InvertibleLayer.
    """
    def _forward(self, x):
        """
        :param x: List of tensors
        :return: Reversed list of tensors
        """
        assert isinstance(x, list)
        return x[::-1]

    def _reverse(self, y):
        """
        :param y: List of tensors
        :return: Reversed list of tensors
        """
        assert isinstance(y, list)
        return y[::-1]


class InvertibleSequential(torch.nn.Sequential,InvertibleModule):
    """
    An sequential model of Invertible modules. No need to implement the forward
    pass, this is already implemented by torch.nn.Sequential
    """
    def __init__(self, *args):
        # Make sure that all modules in the sequence are invertible
        assert all([isinstance(module, InvertibleModule) for module in args])
        super().__init__(*args)

    def reverse(self, input):
        for module in reversed(self._modules.values()):
            input = module.reverse(input)
        return input


# Construct a sequential module with 3 invertible layers
model = InvertibleSequential(ListReverse(),ListReverse(),ListReverse())
# Wrap the model to prepare it for invert to learn
model = MemoryFreeInvertibleModule(model)
# Move the model CUDA if possible
model.to(device)

print('Simple list of Tensors')
x_in = [torch.ones(3, requires_grad=True).to(device), torch.zeros(2,2).to(device)]
print(x_in)

print('\nModel output: list is reversed')
y = model.forward(x_in)
print(y)

print('\nUndoing the reverse, list is same as original again')
x = model.reverse(y)
print(x)

print('\nConfirming the correct gradient')
print(torch.autograd.grad(y[1].mean(),x_in[0]))