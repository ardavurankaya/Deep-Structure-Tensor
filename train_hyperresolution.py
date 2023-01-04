import numpy as np
import torch
from structure_tensor import eig_special_2d, structure_tensor_2d, eig_special_3d, structure_tensor_3d
import skimage.io 
import matplotlib.pyplot as plt
import scipy as scp
from Models.network2_3d import network2_3d
from torch.linalg import norm
from util import utils
from my_classes import Dataset 

"""
Training for the hyperresolution.
"""

if torch.cuda.is_available():
  device = 'cuda:0'
else:
  device = 'cpu'

params = {'batch_size': 4,
          'shuffle': True,
          'num_workers': 2}

training_ids = np.load('data/data_for_3_3d/training_ids.npy')
validation_ids = np.load('data/data_for_3_3d/validation_ids.npy')
test_ids = np.load('data/data_for_3_3d/test_ids.npy')

training_set = Dataset(training_ids)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(validation_ids)
validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=len(validation_ids))

model = network2_3d().to(device)
model.load_state_dict(torch.load('Models3d/Model2_15res_3d.pth'))
optimizer = torch.optim.Adam(model.parameters(), lr=5e-5)

total_epochs = 500
lowest_val_loss = np.inf

epoch = 0
val_loss_tracker = 0

print('Training starts!')

while epoch < total_epochs and val_loss_tracker < 20:
  epoch_loss = 0
  for tensor, interpolation in training_generator:
    tensor, interpolation = tensor.to(device), interpolation.to(device)
    prediction = model.forward(interpolation)
    loss = norm(prediction - tensor) / norm(tensor)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  with torch.no_grad():
    for tensor, interpolation in validation_generator:
      tensor, interpolation = tensor.to(device), interpolation.to(device)
      prediction = model.forward(interpolation)
      val_loss = norm(prediction - tensor) / norm(tensor)
      if val_loss < lowest_val_loss:
        print('New minimum of validation loss is reached')
        lowest_val_loss = val_loss
        val_loss_tracker = 0
        torch.save(model.state_dict(), 'Models3d/Model2_1_15res_3d.pth')
  epoch += 1
  val_loss_tracker += 1
  print(f'Training loss at epoch {epoch} is {epoch_loss/12} and validation loss is {val_loss}')

print('Training finished!')
