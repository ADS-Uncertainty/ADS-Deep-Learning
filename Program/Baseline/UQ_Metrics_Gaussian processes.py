import torch
from LSTM_CSP_model import highwayNet
from LSTM_CSP_util import ngsimDataset,maskedNLL,maskedNLLTest,maskedADETest,maskedMSETest
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time
import os

"""
Tips:
1. Gaussian processes data production
2. Calculate the final data in Program： Gaussian_Processes_Calculation.py
"""


# Metrics
metric = 'std'

args = {}
args['use_cuda'] = True
args['encoder_size'] = 64
args['decoder_size'] = 128
args['in_length'] = 16
args['out_length'] = 25
args['grid_size'] = (13,3)
args['soc_conv_depth'] = 64
args['conv_3x1_depth'] = 16
args['dyn_embedding_size'] = 32
args['input_embedding_size'] = 32
args['num_lat_classes'] = 3
args['num_lon_classes'] = 2
args['use_maneuvers'] = True
args['train_flag'] = False

net = highwayNet(args)

# Replace it with your model file path
net.load_state_dict(torch.load('/home/Models/LSTM_CSP_Optimal_Model.tar'))
net.eval()  # Dropout closed

# Replace it with your data set file path
# data_traj: Vehicle trajectory data   data_tracks: The corresponding dictionary data.
tsSet = ngsimDataset(data_traj="/home/Data/Testset_traj.pt", data_tracks="/home/Data/Testset_tracks.npy")
tsDataloader = DataLoader(tsSet,batch_size=256,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)

if args['use_cuda']:
    net = net.cuda()

# Noise Levels (inch)
# noise_file = ['L_1', 'L_10', 'L_2', 'L_3', 'L_4', 'L_5', 'L_6', 'L_7', 'L_8', 'L_9']
NOISE_LEVEL = [0.328,0.656,0.984,1.312,1.640,1.968,2.297,2.625,2.953,3.281]

for noise_level in NOISE_LEVEL:
    y_all = []
    x_all = []
    TIME = []
    for num in range(10):
        single_x = []
        single_y = []
        st_time = time.time()
        for i, data in enumerate(tsDataloader):
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

            noise = torch.normal(mean=0, std=noise_level, size=[16, nbrs.shape[1], 2])  # L1 0.1
            nbrs = nbrs + noise

            if args['use_cuda']:
                hist = hist.cuda()
                nbrs = nbrs.cuda()
                mask = mask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                fut = fut.cuda()
                op_mask = op_mask.cuda()

            # Forward pass
            fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
            fut_pred_max = torch.zeros_like(fut_pred[0])
            for k in range(lat_pred.shape[0]):
                lat_man = torch.argmax(lat_pred[k, :]).detach()
                lon_man = torch.argmax(lon_pred[k, :]).detach()
                indx = lon_man * 3 + lat_man
                fut_pred_max[:, k, :] = fut_pred[indx][:, k, :]

            # Predict the standard deviation
            MUX = fut_pred_max[:, :, 2]
            MUX = MUX.cpu()
            MUX = MUX.data.numpy()
            MUX = np.round(MUX,4)
            MUX = np.transpose(MUX)

            start_MUX = MUX[0]
            for x in range(MUX.shape[0] - 1):
                templey1 = MUX[x+1]
                start_MUX = np.concatenate((start_MUX, templey1))
            single_x = np.concatenate((single_x, start_MUX),axis=0)

            MUY = fut_pred_max[:, :, 3]
            MUY = MUY.cpu()
            MUY = MUY.data.numpy()
            MUY = np.round(MUY,4)
            MUY = np.transpose(MUY)
            # print(MUY)     # (256, 25)

            start_MUY = MUY[0]
            for y in range(MUY.shape[0] - 1):
                templey1 = MUY[y+1]
                start_MUY = np.concatenate((start_MUY, templey1))
            single_y = np.concatenate((single_y, start_MUY),axis=0)

        x_all.append(single_x)
        y_all.append(single_y)
    x_all = np.transpose(x_all)
    y_all = np.transpose(y_all)

    # ====================================save array====================================
    # Save the file used for calculation
    # Replace it with your target file path
    np.save('/media/TABLE-AU/' + str(noise_level) + '_mux.npy', x_all)
    np.save('/media/TABLE-AU/' + str(noise_level) + '_muy.npy', y_all)
    # Ground Truth traj
    # np.save('/media/TABLE-AU/L_0_x_gtx.npy', gt_x)
    # np.save('/media/TABLE-AU/L_0_y_gty.npy', gt_y)
    # ==================================================================================
