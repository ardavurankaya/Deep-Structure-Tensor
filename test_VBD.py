import numpy as np
import torch
from structure_tensor import eig_special_2d, structure_tensor_2d, eig_special_3d, structure_tensor_3d
import skimage.io 
import matplotlib.pyplot as plt
import scipy as scp
from Models.network3_3d import network3_3d
from torch.linalg import norm
from util import utils
from my_classes import Dataset 
from util import volume, glyph

if torch.cuda.is_available():
  device = 'cuda:0'
else:
  device = 'cpu'

#test_ids = np.load('data/data_for_3_3d/test_ids.npy')
test_ids = np.array([3])

test_set = Dataset(test_ids)
test_generator = torch.utils.data.DataLoader(test_set, batch_size=len(test_ids))

errors_network = np.zeros(len(test_ids))
errors_interpolation = np.zeros(len(test_ids))

num_conv = 3
model = network3_3d(num_conv).to(device)
model.load_state_dict(torch.load('Models3d/model3_3d_1_lr=1e-4_3conv.pth'))

with torch.no_grad():
  for volume_test, tensor, interpolation in test_generator:
    volume_test, tensor, interpolation = volume_test.to(device), tensor.to(device), interpolation.to(device)
    predictions = model(volume_test, interpolation)
    predictions = predictions.detach().cpu().numpy()
    tensor = tensor.detach().cpu().numpy()
    interpolation = interpolation.detach().cpu().numpy()
    #print(utils.compute_rmse(tensor, predictions))
    for i in range(predictions.shape[0]):
      for j in range(6):
        plt.subplot(3, 3, i +1)
        plt.imshow(predictions[0, i, 50])
      errors_network[i] = utils.compute_rmse(tensor[i], predictions[i])
      errors_interpolation[i] = utils.compute_rmse(tensor[i], interpolation[i])
print(errors_network.mean())
print(errors_interpolation.mean())

volume_test = torch.squeeze(volume_test[0]).detach().cpu().numpy() + 54.80844581854907
network = predictions[0]
S = tensor[0]
interpolation = interpolation[0]
print(volume_test.shape)
print(network.shape)
print(S.shape)
print(interpolation.shape)
plt.show()


