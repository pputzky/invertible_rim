"""
This script demonstrates how to build an invertible Unet, wrap it for invert to learn, and
then train it on a binary classification task.

Here, the data set consists of white noise images that are random labeled.
"""

import torch

from irim import InvertibleUnet
from irim import MemoryFreeInvertibleModule

# Use CUDA if devices are available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- Parameters ---
# Working with images, for time series or volumes set to 1 or 3, respectively
conv_nd = 2
# Number of Householder projections for constructing 1x1 convolutions
n_householder = 3
# Number of channels for each layer of the Invertible Unet
n_channels = [3,3,3,3,3]
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

# Construct Invertible Unet
model = InvertibleUnet(n_channels=n_channels,n_hidden=n_hidden,dilations=dilations,
                       conv_nd=conv_nd, n_householder=n_householder)

# Wrap the model for Invert to Learn
model = MemoryFreeInvertibleModule(model)

# Move model to CUDA device if possible
model.to(device)

# Use data parallel if possible
if torch.cuda.device_count() > 1:
    model = torch.nn.DataParallel(model)

optimizer = torch.optim.Adam(model.parameters(), learning_rate)

# Input data drawn form a standard normal
x_in = torch.randn(n_samples,im_channels,*[im_size]*conv_nd, requires_grad=True, device=device)
# Binary labels for each sample
y_in = torch.empty(n_samples,1,*[im_size]*conv_nd, device=device).random_(2)

for i in range(3000):
    optimizer.zero_grad()
    model.zero_grad()

    # Forward computation
    y_est = model.forward(x_in)
    # We use the first channel for prediction
    y_est = y_est[:,:1]
    loss = torch.nn.functional.binary_cross_entropy_with_logits(y_est, y_in)
    loss.backward()

    optimizer.step()

    if i % 100 == 0:
        y_est = (y_est >= 0.).float()
        accuracy = torch.mean((y_est == y_in).float())
        print('Iteration', i, ': loss =',loss.item(), 'accuracy = ', accuracy.item())
