"""
A simple demonstration for training an IRIM as an image denoiser.

This script will lead through the process of defining a gradient function for the IRIM, then
building the RIM model, and how to train the model using invert to learn. This script will utilize
CUDA devices if available.
"""
import torch

from irim import IRIM
from irim import InvertibleUnet
from irim import MemoryFreeInvertibleModule

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Parameters ---
# Working with images, for time series or volumes set to 1 or 3, respectively
conv_nd = 2
# Number of Householder projections for constructing 1x1 convolutions
n_householder = 3
# Number of channels for each layer of the Invertible Unet
n_channels = [16,8,4,8,16]
# Number of hidden channel in the residual functions of the Invertible Unet
n_hidden = [16] * 5
# Downsampling factors
dilations = [1,2,4,2,1]
# Number of IRIM steps
n_steps = 5
# Number of image channels
im_channels = 3
# Number of total samples
n_samples = 64
im_size = 32
learning_rate = 1e-3


def grad_fun(x_est,y):
    """
    Defines the gradient function for a denoising problem with White Noise.

    This function demonstrates the use of  Pytorch's autograd to calculate the gradient.
    In this example, the function is equivalent to

    def grad_fun(x_est,y):
        return x_est - y

    :param x_est: Tensor, model estimate of x
    :param y: Tensor, noisy measurements
    :return: grad_x
    """
    # True during training, False during testing
    does_require_grad = x_est.requires_grad

    with torch.enable_grad():
        # Necessary for using autograd
        x_est.requires_grad_(True)
        # Assuming uniform white noise, in the denoising case matrix A is the identity
        error = torch.sum((y - x_est)**2)
        # We retain the graph during training only
        grad_x = torch.autograd.grad(error, inputs=x_est, retain_graph=does_require_grad,
                                     create_graph=does_require_grad)[0]
    # Set requires_grad back to it's original state
    x_est.requires_grad_(does_require_grad)

    return grad_x

# def grad_fun(x_est, y):
#     return x_est - y
# At every iteration of the IRIM we use an Invertible Unet for processing. Note, that the use of ModuleList
# is necessary for Pytorch to properly register all modules.
step_models = torch.nn.ModuleList([InvertibleUnet(n_channels=n_channels,n_hidden=n_hidden,dilations=dilations,
                                   conv_nd=conv_nd, n_householder=n_householder) for i in range(n_steps)])

# Build IRIM
model = IRIM(step_models,grad_fun,im_channels)
# Wrap the model to be trained with invert to learn
model = MemoryFreeInvertibleModule(model)
model.to(device)
# Use DataParallel if multiple devices are available
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# We generate a simple toy data set where the ground truth data has the same values in the image
# dimensions but different values across batch and channel dimensions. This demonstrates that the
# IRIM can deal with the implicit structure in the data, with a high range of values, and it can even
# do extrapolation.
x = torch.ones(n_samples,im_channels,*[im_size]*conv_nd, requires_grad=False, device=device)
x = torch.cumsum(x,0)
x = torch.cumsum(x,1)
y = x + torch.randn_like(x)

# Training and test split. This will result un an extrapolation problem on the test set.
y, y_test = torch.chunk(y,2,0)
x, x_test = torch.chunk(x,2,0)

# Initial states of the IRIM
x_in = torch.cat((y,torch.zeros(y.size(0),n_channels[0]-im_channels,*[im_size]*conv_nd, device=device)),1)
x_test_in = torch.cat((y_test,torch.zeros(y_test.size(0),n_channels[0]-im_channels,*[im_size]*conv_nd, device=device)),1)
x_in.requires_grad_(True)
x_test_in.requires_grad_(False)

for i in range(3000):
    optimizer.zero_grad()
    model.zero_grad()

    # We only regress on the image dimensions
    x_est = model.forward(x_in, y)[:,:im_channels]
    loss = torch.nn.functional.mse_loss(x_est, x)

    loss.backward()
    optimizer.step()

    if i % 100 == 0:
        model.eval()
        with torch.no_grad():
            x_est = model.forward(x_test_in, y_test)[:, :im_channels]
            loss = torch.nn.functional.mse_loss(x_est, x_test)
            loss_noisy = torch.nn.functional.mse_loss(y_test, x_test)
        print('Iteration', i, ': test loss =',loss.item(), ' loss noisy image =',loss_noisy.item())
        model.train()
