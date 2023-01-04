import torch
from torch import nn
from torch.nn.functional import interpolate
from structure_tensor import structure_tensor_3d

class network3_3d(nn.Module):
  def __init__(self, num_conv):
    super(network3_3d, self).__init__()

    self.conv1 = nn.Conv3d(in_channels = 1, out_channels = 32, kernel_size=5, padding='same')
    self.conv_layers = nn.ModuleList()
    self.num_conv = num_conv
    for k in range(self.num_conv):
      self.conv_layers.append(nn.Conv3d(in_channels=32, out_channels=32, kernel_size=5, padding='same'))
    self.conv2 = nn.Conv3d(in_channels=32, out_channels=6, kernel_size=5, padding='same')
    self.conv3 = nn.Conv3d(in_channels=6, out_channels=6, kernel_size=5, padding='same')
    self.activate = nn.ReLU()
  
  def forward(self, x, up_tensor):

    x = self.activate(self.conv1(x))
    for k in range(self.num_conv):
      f = self.conv_layers[k]
      x = self.activate(f(x))
    x = self.activate(self.conv2(x))
    x =self.conv3(x)
    return x + up_tensor

