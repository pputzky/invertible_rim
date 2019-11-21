import torch
from torch.testing import assert_allclose

from irim.core.invertible_layers import Housholder1x1, RevNetLayer
from irim.test.utils import create_model_and_i2l_copy, forward_reverse, model_gradients

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


def test_householder1x1():
    torch.manual_seed(42)
    x = torch.randn(10, 64, 28, 28)
    y = torch.randn(10, 64, 28, 28)

    model, model_i2l = create_model_and_i2l_copy(Housholder1x1,64,32,2)

    x_est = forward_reverse(model,x)
    gradients_forward, gradients_reverse = model_gradients(model,x,y)
    gradients_forward_i2l, gradients_reverse_i2l = model_gradients(model_i2l,x,y)

    assert_allclose(x_est, x)

    for g_est,g in zip(gradients_forward_i2l,gradients_forward):
        assert_allclose(g_est,g)

    for g_est,g in zip(gradients_reverse_i2l,gradients_reverse):
        assert_allclose(g_est,g)


def test_revnetlayer():
    torch.manual_seed(42)
    x = torch.randn(10, 64, 28, 28)
    y = torch.randn(10, 64, 28, 28)

    model, model_i2l = create_model_and_i2l_copy(RevNetLayer, 64, 32, conv_nd=2)

    x_est = forward_reverse(model,x)
    gradients_forward, gradients_reverse = model_gradients(model,x,y)
    gradients_forward_i2l, gradients_reverse_i2l = model_gradients(model_i2l,x,y)

    assert_allclose(x_est, x)

    for g_est,g in zip(gradients_forward_i2l,gradients_forward):
        assert_allclose(g_est,g)

    for g_est,g in zip(gradients_reverse_i2l,gradients_reverse):
        assert_allclose(g_est,g)
