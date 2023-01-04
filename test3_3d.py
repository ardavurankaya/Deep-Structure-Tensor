import numpy as np
import torch
from structure_tensor import eig_special_2d, structure_tensor_2d, eig_special_3d, structure_tensor_3d
import skimage.io 
import matplotlib.pyplot as plt
import scipy as scp
from network3_3d import network3_3d
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


"""
val0, vec0 = eig_special_3d(S)

# # Fix pole order from ZYX to XYZ
val0 = val0[[2,1,0],:]
vec0 = vec0[[2,1,0],:]

# (Optional) Calculate linearity score - how signifcant the found direction is
# Can be entered as weights parameter in visualization functions
lin = (val0[1]-val0[0])/val0[2]

# (Optional) Flip opposite vectors to face the same direction - easier visualization
flipOpposites = True
flipMask = vec0[0,:] < 0
flipMask = np.array([flipMask,flipMask,flipMask])
vec0[flipMask] = -vec0[flipMask]

# Prepare mask - focus only on bright parts
bgMask = volume_test < 60

# Color representation of the directions in a volume
# rgba = volume.convertToFan(vec, halfSphere=flipOpposites, weights=None, mask=bgMask) #Old method (new name)
rgba1 = volume.convertToIco(vec0, weights=lin, mask=bgMask) #New, bettter color scheme
volume.saveRgbaVolume(rgba1,savePath="data/results/tensor3_volume.tiff")


#%% alternative visualization - direction histogram visualized as a glyph

sph = glyph.orientationVec(vec0.reshape(3,-1), fullSphere=True, weights=None) 

H, el, az, binArea = glyph.histogram2d(sph,bins=[100,200],norm='prob_binArea', weights=None)

glyph.save_glyph(H,el,az,savePath="data/results/tensor3_glyph.vtk", colorMap='Ico', flipColor=flipOpposites)

## DO the same for network output

val1, vec1 = eig_special_3d(network)

# # Fix pole order from ZYX to XYZ
val1 = val1[[2,1,0],:]
vec1 = vec1[[2,1,0],:]

# (Optional) Calculate linearity score - how signifcant the found direction is
# Can be entered as weights parameter in visualization functions
lin = (val1[1]-val1[0])/val1[2]

# (Optional) Flip opposite vectors to face the same direction - easier visualization
flipOpposites = True
flipMask = vec1[0,:] < 0
flipMask = np.array([flipMask,flipMask,flipMask])
vec1[flipMask] = -vec1[flipMask]

# Prepare mask - focus only on bright parts
bgMask = volume_test < 60

# Color representation of the directions in a volume
# rgba = volume.convertToFan(vec, halfSphere=flipOpposites, weights=None, mask=bgMask) #Old method (new name)
rgba = volume.convertToIco(vec1, weights=lin, mask=bgMask) #New, bettter color scheme
volume.saveRgbaVolume(rgba,savePath="data/results/network3_volume.tiff")


#%% alternative visualization - direction histogram visualized as a glyph

sph = glyph.orientationVec(vec1.reshape(3,-1), fullSphere=True, weights=None) 

H, el, az, binArea = glyph.histogram2d(sph,bins=[100,200],norm='prob_binArea', weights=None)

glyph.save_glyph(H,el,az,savePath="data/results/network3_glyph.vtk", colorMap='Ico', flipColor=flipOpposites)

#Do the same for interpolation

val1, vec1 = eig_special_3d(interpolation)

# # Fix pole order from ZYX to XYZ
val1 = val1[[2,1,0],:]
vec1 = vec1[[2,1,0],:]

# (Optional) Calculate linearity score - how signifcant the found direction is
# Can be entered as weights parameter in visualization functions
lin = (val1[1]-val1[0])/val1[2]

# (Optional) Flip opposite vectors to face the same direction - easier visualization
flipOpposites = True
flipMask = vec1[0,:] < 0
flipMask = np.array([flipMask,flipMask,flipMask])
vec1[flipMask] = -vec1[flipMask]

# Prepare mask - focus only on bright parts
bgMask = volume_test < 60

# Color representation of the directions in a volume
# rgba = volume.convertToFan(vec, halfSphere=flipOpposites, weights=None, mask=bgMask) #Old method (new name)
rgba = volume.convertToIco(vec1, weights=lin, mask=bgMask) #New, bettter color scheme
volume.saveRgbaVolume(rgba,savePath="data/results/interpolation3_volume.tiff")


#%% alternative visualization - direction histogram visualized as a glyph

sph = glyph.orientationVec(vec1.reshape(3,-1), fullSphere=True, weights=None) 

H, el, az, binArea = glyph.histogram2d(sph,bins=[100,200],norm='prob_binArea', weights=None)

glyph.save_glyph(H,el,az,savePath="data/results/interpolation3_glyph.vtk", colorMap='Ico', flipColor=flipOpposites)

"""
