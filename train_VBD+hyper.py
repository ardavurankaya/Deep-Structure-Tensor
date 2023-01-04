import numpy as np
import torch
from structure_tensor import eig_special_2d, structure_tensor_2d, eig_special_3d, structure_tensor_3d
import skimage.io 
import matplotlib.pyplot as plt
import scipy as scp
from models.network3_3d import network3_3d
from torch.linalg import norm
from util import utils
from my_classes import Dataset 

if torch.cuda.is_available():
  device = 'cuda:0'
else:
  device = 'cpu'

params = {'batch_size': 8,
          'shuffle': True,
          'num_workers': 2}

training_ids = np.load('data/data_for_3_3d/training_ids.npy')
validation_ids = np.load('data/data_for_3_3d/validation_ids.npy')
test_ids = np.load('data/data_for_3_3d/test_ids.npy')

training_set = Dataset(training_ids)
training_generator = torch.utils.data.DataLoader(training_set, **params)

validation_set = Dataset(validation_ids)
validation_generator = torch.utils.data.DataLoader(validation_set, batch_size=len(validation_ids))

num_conv = 3
model = network3_3d(num_conv).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

total_epochs = 500
val_loss_tracker = 0
epoch = 0
lowest_val_loss = np.inf

print('Training starts!')

while epoch < total_epochs and val_loss_tracker < 20:
  epoch_loss = 0
  for local_volumes, local_tensors, local_interpolations in training_generator:
    local_volumes, local_tensors, local_interpolations = local_volumes.to(device), local_tensors.to(device), local_interpolations.to(device)
    prediction = model.forward(local_volumes, local_interpolations)
    loss = norm(prediction - local_tensors) / norm(local_tensors)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    epoch_loss += loss.item()
  with torch.no_grad():
    for volume, tensor, interpolation in validation_generator:
      volume, tensor, interpolation = volume.to(device), tensor.to(device), interpolation.to(device)
      prediction = model.forward(volume, interpolation)
      val_loss = norm(prediction - tensor) / norm(tensor)
      if val_loss < lowest_val_loss:
        print("New minimum of validation loss is reached!")
        lowest_val_loss = val_loss
        val_loss_tracker = 0
        torch.save(model.state_dict(), 'Models3d/model4_3d_lr=1e-4_3conv.pth')
  epoch += 1
  val_loss_tracker = 0
  print(f"Training loss at epoch {epoch} is {epoch_loss / 12} and validation_loss is {val_loss}")

print('Training finished!')

