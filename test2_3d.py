import numpy as np
import torch
from structure_tensor import eig_special_2d, structure_tensor_2d, eig_special_3d, structure_tensor_3d
import skimage.io 
import matplotlib.pyplot as plt
import scipy as scp
from network2_3d import network2_3d
from torch.linalg import norm
from util import utils
from my_classes import Dataset 

if torch.cuda.is_available():
  device = 'cuda:0'
else:
  device = 'cpu'

test_ids = np.load('data/data_for_3_3d/validation_ids.npy')

test_set = Dataset(test_ids)
test_generator = torch.utils.data.DataLoader(test_set, batch_size=len(test_ids))

errors_network = np.zeros(len(test_ids))

model = network2_3d().to(device)
model.load_state_dict(torch.load('Models3d/Model2_1_15res_3d.pth'))

with torch.no_grad():
  for tensor, interpolation in test_generator:
    tensor, interpolation = tensor.to(device), interpolation.to(device)
    predictions = model(interpolation)
    predictions = predictions.detach().cpu().numpy()
    tensor = tensor.detach().cpu().numpy()
    interpolation = interpolation.detach().cpu().numpy()
    #print(utils.compute_rmse(tensor, predictions))
    for i in range(predictions.shape[0]):
      errors_network[i] = utils.compute_rmse(tensor[i], predictions[i])
      #errors_interpolation[i] = utils.compute_rmse(tensor[i], interpolation[i])
print(errors_network.mean())
#print(errors_interpolation.mean())