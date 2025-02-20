import numpy as np
import pandas as pd
# np.set_printoptions(precision=4)

"""
Calculate AU by Gaussian Processes
"""

# Replace it with your target file path
# NOISE_LEVEL = [0.328,0.656,0.984,1.312,1.640,1.968,2.297,2.625,2.953,3.281]
x_all = np.load('/media/TABLE-AU/1.64/1.64_mux.npy')
y_all = np.load('/media/TABLE-AU/1.64/1.64_muy.npy')
# gt_y = np.load('/media/chengjie/16AAE3CCAAE3A689/MCDARRAY/L_0_y_gty.npy')

print('Data in!')
print(x_all.shape)

x_all = np.transpose(x_all)
y_all = np.transpose(y_all)

X = []
Y = []

for i in range(10):

    x_mean = np.mean(1/x_all[i])
    y_mean = np.mean(1/y_all[i])
    X.append(x_mean)
    Y.append(y_mean)

print('X:',X)
print('Y:',Y)

X_mean = np.mean(X)
Y_mean = np.mean(Y)

print('x_mean:', X_mean)
print('y_mean:', Y_mean)
all_mean = (X_mean + Y_mean)/2
print('ALL_mean:', all_mean)

x_m = np.std(X,ddof=1)
y_m = np.std(Y,ddof=1)

print('x_STD:', x_m)
print('y_STD:', y_m)
all_STD = (x_m + y_m)/2
print('ALL_STD:', all_STD)

# ====================================================================================
# # Original

# x_all = np.load('/media/TABLE-AU/base/base_x.npy')
# y_all = np.load('/media/TABLE-AU/base/base_y.npy')
# # gt_y = np.load('/media/TABLE-AU/L_0_y_gty.npy')

# print('Data in!')
# print(x_all)
# print(y_all)
#
# # x_all = np.transpose(x_all)
# # y_all = np.transpose(y_all)
# #
# x_mean = np.mean(1/x_all)
# y_mean = np.mean(1/y_all)
#
# print('x_mean:', x_mean)
# print('y_mean:', y_mean)
# all_mean = (x_mean + y_mean)/2
# print('ALL_mean:', all_mean)
# ====================================================================================