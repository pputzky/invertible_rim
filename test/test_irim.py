import torch
from torch.testing import assert_allclose

from irim.invertible_unet import InvertibleUnet
from irim.irim import IRIM, InvertibleGradUpdate
from test.utils import create_model_and_i2l_copy, forward_reverse, model_gradients

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


def test_invertiblegradupdate():
    torch.manual_seed(42)

    def grad_fun(x, z):
        return x - z

    x = torch.randn(10, 64, 28, 28)
    y = torch.randn(10, 64, 28, 28)
    z = torch.randn(10, 16, 28, 28)

    model, model_i2l = create_model_and_i2l_copy(InvertibleGradUpdate, grad_fun, z.size(1))

    x_est = forward_reverse(model,x,z)
    gradients_forward, gradients_reverse = model_gradients(model,x,y,z)
    gradients_forward_i2l, gradients_reverse_i2l = model_gradients(model_i2l,x,y,z)

    assert_allclose(x_est, x)

    for g_est,g in zip(gradients_forward_i2l,gradients_forward):
        assert_allclose(g_est,g)

    for g_est,g in zip(gradients_reverse_i2l,gradients_reverse):
        assert_allclose(g_est,g)


def test_irim():
    torch.manual_seed(42)

    def grad_fun(x, z):
        grad = (x - z)
        grad = grad / x.norm(2,dim=(-2,-1),keepdim=True)
        return grad

    x = torch.randn(10, 64, 28, 28)
    y = torch.randn(10, 64, 28, 28)
    z = torch.randn(10, 16, 28, 28)

    def get_model():
        unets = torch.nn.ModuleList([InvertibleUnet([64,32,16], [32,8,24], [1,2,4])]*10)
        model = IRIM(unets, grad_fun, n_channels=z.size(1))
        return model

    model, model_i2l = create_model_and_i2l_copy(get_model)

    x_est = forward_reverse(model,x,z)
    gradients_forward, gradients_reverse = model_gradients(model,x,y,z)
    gradients_forward_i2l, gradients_reverse_i2l = model_gradients(model_i2l,x,y,z)

    assert_allclose(x_est, x, rtol=1e-3, atol=1e-4)

    for g_est,g in zip(gradients_forward_i2l,gradients_forward):
        assert_allclose(g_est,g)

    for g_est,g in zip(gradients_reverse_i2l,gradients_reverse):
        assert_allclose(g_est,g)
