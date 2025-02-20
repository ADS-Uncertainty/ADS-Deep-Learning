import numpy as np
import pandas as pd

"""
Calculate EU by Deep ensembles
"""

# Replace it with your target file path
x_all = np.load('/media/TABLE-DE/noise_level_mux.npy')
y_all = np.load('/media/TABLE-DE/noise_level_muy.npy')
# Ground Truth traj
gt_y = np.load('/media/TABLE-DE/L_0_y_gty.npy')

point = np.where(gt_y == 0)
print('Data in!')
print(x_all.shape)
print(point[0].shape)

mask = np.ones(x_all.shape[0],dtype=bool)
mask[point[0]] = False
x_all1 = x_all[mask]
y_all1 = y_all[mask]

print(x_all1.shape)

x_all1 = np.transpose(x_all1)
y_all1 = np.transpose(y_all1)
# x_all = np.array([x_all[i] for i in range(x_all.shape[0]) if i not in point1[0]])

# print(y_all)
x_std = np.std(x_all1,axis=0,ddof=1)
x_std_mean = np.mean(x_std)
y_std = np.std(y_all1,axis=0,ddof=1)
y_std_mean = np.mean(y_std)

print('X_STD:', x_std_mean)
print('Y_STD:', y_std_mean)
all_std_mean = (x_std_mean + y_std_mean)/2
print('ALL_STD:', all_std_mean)