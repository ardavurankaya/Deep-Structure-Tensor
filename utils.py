import numpy as np
import torch
from structure_tensor import eig_special_2d, structure_tensor_2d, eig_special_3d, structure_tensor_3d  #https://github.com/Skielex/structure-tensor
import skimage.io 
import matplotlib.pyplot as plt
import scipy as scp
import cv2
import torch
import time
from torch.nn.functional import interpolate

def compute_rmse(ref, pred):
  difference = (ref - pred) ** 2
  error = np.sum(difference)
  error = error / np.sum(ref ** 2)
  return np.sqrt(error)


def generate_volume(I, cube_size):
  """
  Takes original 3D volume(I) and divides data into smaller cubes denoted by cube_size
  Also saves the created chunks as torch tensors
  """
  ind = 0 # data id
  I -= I.mean()
  print('Creating volumes...')
  for i in range(int(I.shape[0] / cube_size)):
    for j in range(int(I.shape[1] / cube_size)):
      for k in range(int(I.shape[2] / cube_size)):
        temp = I[i*cube_size:(i+1)*cube_size, j*cube_size:(j+1)*cube_size, k*cube_size:(k+1)*cube_size]
        #temp = torch.from_numpy(temp)
        np.save('data/data_for_3_3d/volume' + str(ind), temp)
        ind += 1
  print('Creating volumes finished!')

def generate_structure_tensor(I, cube_size):
  ind = 0
  rho = 8
  sigma = rho / 2
  t0 = time.time()
  print('Creating structure_ tensors...')
#  S = structure_tensor_3d(I, rho, sigma)
  for i in range(int(I.shape[0] / cube_size)):
    for j in range(int(I.shape[1] / cube_size)):
      for k in range(int(I.shape[2] / cube_size)):
        temp = I[i*cube_size:(i+1)*cube_size, j*cube_size:(j+1)*cube_size, k*cube_size:(k+1)*cube_size]
        #temp = torch.from_numpy(temp)
        S = structure_tensor_3d(temp, rho, sigma)
        np.save('data/data_for_3_3d/structure_tensor' + str(ind), S)
        ind += 1  
  print('Creating structure tensors finished!')

def generate_interpolated_tensor(I, cube_size):
  ind = 0
  rho = 8
  sigma = rho / 2
  t0 = time.time()
  print('Create interpolated tensors...')
  t1 = time.time()
  for i in range(int(I.shape[0] / cube_size)):
    for j in range(int(I.shape[1] / cube_size)):
      for k in range(int(I.shape[2] / cube_size)):
        temp = I[i*cube_size:(i+1)*cube_size, j*cube_size:(j+1)*cube_size, k*cube_size:(k+1)*cube_size]
        temp = torch.unsqueeze(torch.unsqueeze(torch.from_numpy(temp), 0), 0)
        temp = interpolate(temp, scale_factor=(0.5, 0.5, 0.5))
        temp = torch.squeeze(temp)
        S = structure_tensor_3d(temp, rho, sigma)
        S = torch.unsqueeze(torch.from_numpy(S), 0)
        S = interpolate(S, scale_factor=(2,2,2), mode='trilinear')
        S = torch.squeeze(S).numpy()
        #temp = torch.from_numpy(temp)
        np.save('data/data_for_3_3d/structure_tensor_interpolated' + str(ind), S)
        ind += 1  
  print(t1-t0)
  print('Creating interpolated structure tensors finished!')



