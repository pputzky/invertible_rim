import torch
from torch.testing import assert_allclose

from irim import InvertibleUnet
from irim.test.utils import create_model_and_i2l_copy, forward_reverse, model_gradients

if torch.cuda.is_available():
    torch.set_default_tensor_type(torch.cuda.FloatTensor)


def test_invertibleunet():
    torch.manual_seed(42)
    x = torch.randn(10,64,28,28)
    y = torch.randn(10,64,28,28)

    depth = 10
    model, model_i2l = create_model_and_i2l_copy(InvertibleUnet,[64,32,16]*depth, [32,8,24] * depth, [1,2,4] * depth)

    x_est = forward_reverse(model,x)
    gradients_forward, gradients_reverse = model_gradients(model,x,y)
    gradients_forward_i2l, gradients_reverse_i2l = model_gradients(model_i2l,x,y)

    assert_allclose(x_est, x)

    for g_est,g in zip(gradients_forward_i2l,gradients_forward):
        assert_allclose(g_est,g)

    for g_est,g in zip(gradients_reverse_i2l,gradients_reverse):
        assert_allclose(g_est,g)
