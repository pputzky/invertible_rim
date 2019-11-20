from abc import ABC, abstractmethod
import itertools

import torch
from torch import nn
from torch.autograd import Function


class InvertToLearnFunction(Function):
    @staticmethod
    def forward(ctx, n_args, layer, mode, args, kwargs, *tensors):
        if n_args == 1:
            x = tensors[0]
        else:
            x = list(tensors[:2 * n_args:2])

        with torch.no_grad():
            if mode == 'forward':
                y = layer._forward(x, *args, **kwargs)
            elif mode == 'reverse':
                y = layer._reverse(x, *args, **kwargs)

        if any([x_.requires_grad for x_ in tensors]):
            ctx.n_args = n_args
            ctx.layer = layer
            ctx.mode = mode
            ctx.parameters = list(tensors[2 * n_args:])
            ctx.args = args
            ctx.kwargs = kwargs
            if layer.save_input:
                if n_args == 1:
                    ctx.save_for_backward(x)
                else:
                    ctx.save_for_backward(*x)

        if isinstance(y, list):
            y = [(y_, y_.detach()) for y_ in y]
            y = tuple(itertools.chain(*y))
        else:
            y = (y, y.detach())

        return y

    @staticmethod
    def backward(ctx, *out):
        if len(out) == 2:
            y, grad_outputs = out[1], out[0]
        else:
            y, grad_outputs = list(out[1::2]), list(out[::2])

        parameters = ctx.parameters

        x = None
        if len(ctx.saved_tensors) > 0:
            x = list(ctx.saved_tensors)
            if ctx.n_args == 1:
                x = x[0]

        if ctx.mode == 'forward':
            forward_fun = ctx.layer._forward
            reverse_fun = ctx.layer._reverse
        elif ctx.mode == 'reverse':
            forward_fun = ctx.layer._reverse
            reverse_fun = ctx.layer._forward

        x, grad_inputs, param_grads = ctx.layer.gradfun(forward_fun, reverse_fun,
                                                        x, y, grad_outputs, parameters,
                                                        *ctx.args, **ctx.kwargs)
        if ctx.n_args == 1:
            input_gradients = (grad_inputs, x)
        else:
            input_gradients = tuple(itertools.chain(*zip(grad_inputs, x)))
        parameter_gradients = tuple(param_grads)

        # References need to be cleared to prevent memory leaks
        ctx.n_args = None
        ctx.layer = None
        ctx.mode = None
        ctx.parameters = None
        ctx.args = None
        ctx.kwargs = None

        return (None, None, None, None, None) + input_gradients + parameter_gradients


class InvertibleModule(nn.Module, ABC):
    """
    Abstract class to define any invertible Module, be it a layer or whole network.
    Inheriting class should implement forward and backward.
    """
    @abstractmethod
    def forward(self, x, *args, **kwargs):
        pass

    @abstractmethod
    def reverse(self, y, *args, **kwargs):
        pass


class InvertibleLayer(InvertibleModule, ABC):
    """
    Abstract class for all invertible layers. This builds the core of all invertible newtworks.
    All sub-classes are required to implement _forward and _backward which implement the layers
    computations in the respective directions.
    """
    def __init__(self):
        super().__init__()
        self.memory_free = False
        self.save_input = False

    def forward(self, x, *args, **kwargs):
        """
        Forward operation of the invertible layer
        :param x: Tensor or list of Tensors. Only gradients for x will be valid in invert to learn.
        :param args: Additional Inputs
        :param kwargs: Keyword Arguments
        :return: Tensor or list of Tensors
        """
        if self.memory_free:
            if isinstance(x, list):
                n_args = len(x)
                x = list(itertools.chain(*x))
            else:
                n_args = 1
                x = list(x)
            tensors = x + list(self.parameters())
            y = InvertToLearnFunction.apply(n_args, self, 'forward',
                                            args, kwargs, *tensors)
            if len(y) > 2:
                y = list(zip(y[::2], y[1::2]))
        else:
            y = self._forward(x, *args, **kwargs)

        return y

    def reverse(self, y, *args, **kwargs):
        """
        Reverse operation of the invertible layer
        :param y: Tensor or list of Tensors. Only gradients for y will be valid in invert to learn.
        :param args: Additional Inputs
        :param kwargs: Keyword Arguments
        :return: Tensor or list of Tensors
        """
        if self.memory_free:
            if isinstance(y, list):
                n_args = len(y)
                y = list(itertools.chain(*y))
            else:
                n_args = 1
                y = list(y)

            tensors = list(y) + list(self.parameters())
            x = InvertToLearnFunction.apply(n_args, self, 'reverse',
                                            args, kwargs, *tensors)
            if len(x) > 2:
                x = list(zip(x[::2], x[1::2]))
        else:
            x = self._reverse(y, *args, **kwargs)
        return x

    def gradfun(self, forward_fun, reverse_fun, x=None, y=None, grad_outputs=None, parameters=None, *args, **kwargs):
        """
        This function implements gradient calculations for the invert to learn case. It will be called by
        InvertToLearn.backward. This function will be valid for any invertible function, however computation
        cost might be not optimal. Inheriting classes can overwrite this method to implement more efficient
        gradient computation specific for teh respective layer. See invertible_layer.py for examples.
        :param forward_fun: The function that was used during the forward operation
        :param reverse_fun: The inverse of forward_fun
        :param x: Tensor or list of Tensors, Input of layer
        :param y: Tensor or list of Tensors, Output of layer
        :param grad_outputs: Tensor or list of Tensors, gradients passed from higher layers
        :param parameters: Tensor or list of Tensors, parameters of the layer
        :return: x, grad_x, grads_param
        """
        assert not (x is None and y is None)
        if x is None:
            with torch.no_grad():
                x = reverse_fun(y, *args, **kwargs)

        with torch.enable_grad():
            if isinstance(x, list):
                x = [x_.detach().requires_grad_(True) for x_ in x]
                grad_tensors = x + parameters
            else:
                x = x.detach().requires_grad_(True)
                grad_tensors = [x] + parameters
            y = forward_fun(x, *args, **kwargs)
            grads = torch.autograd.grad(y, grad_tensors, grad_outputs=grad_outputs)

        if isinstance(x, list):
            grad_inputs = grads[:len(x)]
            grads_param = grads[len(x):]
        else:
            grad_inputs = grads[0]
            grads_param = grads[1:]

        return x, grad_inputs, grads_param

    @abstractmethod
    def _forward(self, x, *args, **kwargs):
        pass

    @abstractmethod
    def _reverse(self, y, *args, **kwargs):
        pass


class IdentityLayer(InvertibleLayer):
    def _forward(self, x, *args, **kwargs):
        return x

    def _reverse(self, y, *args, **kwargs):
        return y


def make_layer_memory_free(layer, save_input=False):
    if isinstance(layer, InvertibleLayer):
        layer.memory_free = True
        layer.save_input = save_input


class MemoryFreeInvertibleModule(InvertibleModule):
    """
    A wrapper class that turns an invertible module into a module that utilizes invert to
    learn during training, i.e. removing intermediate memory storage for back-propagation.
    """
    def __init__(self, model):
        """
        :param model: Model to be wrapped.
        """
        super().__init__()
        assert isinstance(model, InvertibleModule)
        self.model = model.apply(make_layer_memory_free)
        self.save_layer = IdentityLayer()
        make_layer_memory_free(self.save_layer, save_input=True)

    def forward(self, x, *args, **kwargs):
        if isinstance(x, list):
            x = [(x_, x_.detach()) for x_ in x]
        else:
            x = (x, x.detach())
        x = self.model.forward(x, *args, **kwargs)
        x = self.save_layer.forward(x)
        if isinstance(x, list):
            x = [x_[0] for x_ in x]
        else:
            x = x[0]

        return x

    def reverse(self, y, *args, **kwargs):
        if isinstance(y, list):
            y = [(y_, y_.detach()) for y_ in y]
        else:
            y = (y, y.detach())
        y = self.model.reverse(y, *args, **kwargs)
        y = self.save_layer.reverse(y)
        if isinstance(y, list):
            y = [y_[0] for y_ in y]
        else:
            y = y[0]

        return y
