import torch
from torch import nn

class network2_3d(nn.Module):
  def __init__(self):
    super(network2_3d, self).__init__()

    self.conv1 = nn.Conv3d(in_channels=6, out_channels=16, kernel_size=3, padding= 'same')
    self.conv2 = nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding='same')
    self.conv3 = nn.Conv3d(in_channels=16, out_channels=6, kernel_size=3, padding='same')
    self.resnet_layer1 = nn.ModuleList()
    for k in range(15):
      self.resnet_layer1.append(nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding='same'))
    self.resnet_layer2 = nn.ModuleList()
    for k in range(15):
      self.resnet_layer2.append(nn.Conv3d(in_channels=16, out_channels=16, kernel_size=3, padding='same'))
    self.activate = nn.ReLU()
    self.scalar = torch.tensor([0.1], dtype=torch.float32).to('cuda:0')
  
  def forward(self, x):
    
    input_tensor = x
    x = self.conv1(x)
    for k in range(15):
      first_layer = x
      f = self.resnet_layer1[k]
      x = self.activate(f(x))
      f = self.resnet_layer2[k]
      x = f(x)
      x = x * self.scalar
      x = x + first_layer
    x = self.activate(self.conv2(x))
    x = self.conv3(x)
    x = x + input_tensor
    return x