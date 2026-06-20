import numpy as np
from scipy.interpolate import interp1d
import scipy.io as sio
import torch


data = sio.loadmat('SRF/P_N_V2.mat')
data = torch.from_numpy(data['P_20N'])

x_original = np.linspace(0, 1, 31)

x_new = np.linspace(0, 1, 144)


interpolated_matrix = []
for column in data:

    col_np = column.numpy()
    f = interp1d(x_original, col_np, kind='linear')
    interpolated_column = f(x_new)
    interpolated_matrix.append(interpolated_column)

interpolated_matrix = np.array(interpolated_matrix)

sio.savemat('SRF/144.mat', {'srf': interpolated_matrix})
