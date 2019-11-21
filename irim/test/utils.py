import torch
from irim.core.invert_to_learn import InvertibleModule, MemoryFreeInvertibleModule


def forward_reverse(model, x, *args, **kwargs):
    assert isinstance(model, InvertibleModule)
    with torch.no_grad():
        y = model.forward(x, *args, **kwargs)
        x_est = model.reverse(y, *args, **kwargs)

    return x_est


def model_gradients(model, x, y, *args, **kwargs):
    assert isinstance(model, InvertibleModule)

    with torch.enable_grad():
        x.detach_().requires_grad_(True)
        y.detach_()
        y_est = model.forward(x, *args, **kwargs)
        loss = torch.nn.functional.mse_loss(y_est,y)
        grad_forward = torch.autograd.grad(loss, [x] + list(model.parameters()))

    with torch.enable_grad():
        y.detach_().requires_grad_(True)
        x.detach_()
        x_est = model.reverse(y, *args, **kwargs)
        loss = torch.nn.functional.mse_loss(x_est,x)
        grad_reverse = torch.autograd.grad(loss, [y] + list(model.parameters()))

    return grad_forward, grad_reverse


def create_model_and_i2l_copy(module_class, *args, **kwargs):
    model = module_class(*args, **kwargs)
    assert isinstance(model, InvertibleModule)
    model_i2l = module_class(*args, **kwargs)
    model_i2l.load_state_dict(model.state_dict())
    model_i2l = MemoryFreeInvertibleModule(model_i2l)

    return model, model_i2l
