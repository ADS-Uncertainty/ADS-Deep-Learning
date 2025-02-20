import torch
from LSTM_CSP_model import highwayNet
from LSTM_CSP_util import ngsimDataset,maskedNLL,maskedNLLTest,maskedADETest,maskedMSETest
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time
import os
# np.set_printoptions(suppress=True)
"""
Tips:
1. Deep Ensembles data production
2. Calculate the final data in Program： Deep_Ensembles_Calculation.py
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

# Replace it with your target file path
target_files = os.listdir(r"/media/MODEL")
target_folder1 = str("/media/MODEL/")

models = []
for file in target_files:
    models.append(file)
    models.sort()

# Replace it with your fixed noise file path
target_NOISE = os.listdir(r"/media/NOISE")
noise_file = []    # ['L_1', 'L_10', 'L_2', 'L_3', 'L_4', 'L_5', 'L_6', 'L_7', 'L_8', 'L_9']
for file in target_NOISE:
    noise_file.append(file)
    noise_file.sort()


# Replace it with your data set file path
# data_traj: Vehicle trajectory data   data_tracks: The corresponding dictionary data.
tsSet = ngsimDataset(data_traj="/home/Data/Testset_traj.pt", data_tracks="/home/Data/Testset_tracks.npy")
tsDataloader = DataLoader(tsSet,batch_size=256,shuffle=False,num_workers=8,collate_fn=tsSet.collate_fn)


for noise_level in noise_file:
    target_folder = str("/media/NOISE/"+ str(noise_level) + '/')
    print(target_folder)
    y_all = []
    x_all = []
    TIME = []
    for model in models:
        # 10 Models
        net = highwayNet(args)
        net.load_state_dict(torch.load(target_folder1 + model))
        net.eval()  # Dropout closed

        if args['use_cuda']:
            net = net.cuda()

        single_x = []
        single_y = []
        st_time = time.time()
        for i, data in enumerate(tsDataloader):
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

            # Fixed Gaussian white noise
            noise_file = target_folder + f'noise_{i}.pt'
            noise = torch.load(noise_file)
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

            # =======================================MUX & MUY saving=======================================
            MUX = fut_pred_max[:, :, 0]
            MUX = MUX.cpu()
            MUX = MUX.data.numpy()
            MUX = np.round(MUX,4)
            MUX = np.transpose(MUX)

            start_MUX = MUX[0]
            for x in range(MUX.shape[0] - 1):
                templey1 = MUX[x+1]
                start_MUX = np.concatenate((start_MUX, templey1))
            single_x = np.concatenate((single_x, start_MUX),axis=0)

            MUY = fut_pred_max[:, :, 1]
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
    np.save('/media/TABLE-DE/'+ str(noise_level) + '/' + str(noise_level) + '_mux.npy', x_all)
    np.save('/media/TABLE-DE/'+ str(noise_level) + '/' + str(noise_level) + '_muy.npy', y_all)
    # Ground Truth traj
    # np.save('/media/TABLE-DE/L_0_x_gtx.npy', gt_x)
    # np.save('/media/TABLE-DE/L_0_y_gty.npy', gt_y)
    # ==================================================================================