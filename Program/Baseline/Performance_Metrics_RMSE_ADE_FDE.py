from __future__ import print_function
import torch
from LSTM_CSP_model import highwayNet
from LSTM_CSP_util import ngsimDataset,maskedNLL,maskedNLLTest,maskedADETest,maskedMSETest
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import time
import os

np.set_printoptions(suppress=True)

metric = 'SingleMetrics'   # Rmse & ADE & FDE
# metric = 'nll'

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

# Replace it with your data set file path
# data_traj: Vehicle trajectory data   data_tracks: The corresponding dictionary data.
tsSet = ngsimDataset(data_traj="/home/Data/Testset_traj.pt", data_tracks="/home/Data/Testset_tracks.npy")
tsDataloader = DataLoader(tsSet,batch_size=256,shuffle=True,num_workers=8,collate_fn=tsSet.collate_fn)

# Noise Levels: L1-L10(inch)
NOISE_LEVEL = [0.328,0.656,0.984,1.312,1.640,1.968,2.297,2.625,2.953,3.281]

for level in NOISE_LEVEL:
    RMSE = []
    ADE = []
    FDE = []
    NLL = []
    TIME = []
    for x in range(10):
        net.eval()  # Dropout control
        if args['use_cuda']:
            net = net.cuda()

        lossVals = torch.zeros(25).cuda()
        counts = torch.zeros(25).cuda()
        lossVals1 = torch.zeros(25).cuda()
        counts1 = torch.zeros(25).cuda()
        start_time = time.time()

        for i, data in enumerate(tsDataloader):
            # print('New data begin')
            st_time = time.time()
            hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data
            # # =================================
            # # stop shuffle
            # noise_file = f'noise_{i}.pt'
            # noise = torch.load(noise_file)
            # # =================================

            # Add random Gaussian white noise to the trajectory of Surrounding Vehicles
            # When obtaining Original data (Without Noise), please annotate the NOISE_LEVEL and the following two lines
            noise = torch.normal(mean=0, std=level, size=[16, nbrs.shape[1], 2])
            nbrs = nbrs + noise

            # print('=====================================================================================================================')

            # NS.append(nbrs.shape[1])
            # Initialize Variables
            if args['use_cuda']:
                hist = hist.cuda()
                nbrs = nbrs.cuda()
                mask = mask.cuda()
                lat_enc = lat_enc.cuda()
                lon_enc = lon_enc.cuda()
                fut = fut.cuda()
                op_mask = op_mask.cuda()

            if metric == 'nll':
                # Forward pass
                if args['use_maneuvers']:
                    fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l,c = maskedNLLTest(fut_pred, lat_pred, lon_pred, fut, op_mask)

                    lossVals += l.detach()
                    counts += c.detach()
                else:
                    fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l, c = maskedNLLTest(fut_pred, 0, 0, fut, op_mask,use_maneuvers=False)
            else:
                # Forward pass
                if args['use_maneuvers']:
                    fut_pred, lat_pred, lon_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    fut_pred_max = torch.zeros_like(fut_pred[0])
                    for k in range(lat_pred.shape[0]):
                        lat_man = torch.argmax(lat_pred[k, :]).detach()
                        lon_man = torch.argmax(lon_pred[k, :]).detach()
                        indx = lon_man*3 + lat_man
                        fut_pred_max[:,k,:] = fut_pred[indx][:,k,:]
                    l, c = maskedMSETest(fut_pred_max, fut, op_mask)
                    l1, c1 = maskedADETest(fut_pred_max, fut, op_mask)

                    lossVals += l.detach()
                    counts += c.detach()
                    lossVals1 += l1.detach()
                    counts1 += c1.detach()
                else:
                    fut_pred = net(hist, nbrs, mask, lat_enc, lon_enc)
                    l, c = maskedMSETest(fut_pred, fut, op_mask)

        if metric == 'nll':
            nll = torch.sum(lossVals / counts) / 25
            NLL.append(float(nll))
            print('| nll(m)：', ' % .4f' % float(nll), ' |')

        else:
            rmse = torch.sum(torch.pow(lossVals / counts, 0.5)) / 25 * 0.3048
            ade = torch.sum(lossVals1 / counts1) / 25 * 0.3048  # ADE
            fde = lossVals1[-1] / counts1[-1] * 0.3048  # FDE
            print('| rmse(m)：', '%.4f' % float(rmse), '| ade(m)：', '%.4f' % float(ade), ' | fde(m)：', '%.4f' % float(fde))
            RMSE.append(float(rmse))
            ADE.append(float(ade))
            FDE.append(float(fde))

            end_time = time.time()
            Test_time = round(end_time - start_time, 4)
            TIME.append(Test_time)
