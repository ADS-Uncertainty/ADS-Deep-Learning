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
'''
Tips：
1. I: used for counting
2. NS: Record the size of the corresponding neighboring vehicle (nbr) data based on the batch size
'''

metric = 'nll'

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


# Replace it with your model file path
target_files = os.listdir(r"/media/MODEL")
target_folder1 = str("/media/MODEL/")
result = pd.DataFrame()
start_1 = pd.DataFrame()

models = []
for file in target_files:
    models.append(file)
    models.sort()

net = highwayNet(args)

# Replace it with your model file path
net.load_state_dict(torch.load('/media/MODEL/LSTM_CSP_Optimal_Model.tar'))

# Replace it with your data set file path
# data_traj: Vehicle trajectory data   data_tracks: The corresponding dictionary data.
tsSet = ngsimDataset(data_traj="/home/Data/Testset_traj.pt", data_tracks="/home/Data/Testset_tracks.npy")
tsDataloader = DataLoader(tsSet,batch_size=256,shuffle=False,num_workers=8,collate_fn=tsSet.collate_fn)

net.eval()
if args['use_cuda']:
    net = net.cuda()

lossVals = torch.zeros(25).cuda()
counts = torch.zeros(25).cuda()
lossVals1 = torch.zeros(25).cuda()
counts1 = torch.zeros(25).cuda()

I = []
NS = []

for i, data in enumerate(tsDataloader):
    # print('New data begin')
    st_time = time.time()
    hist, nbrs, mask, lat_enc, lon_enc, fut, op_mask = data

    # noise_file = f'noise_{i}.pt'
    # noise = torch.load(noise_file)
    # nbrs = nbrs + noise

    I.append(i)
    NS.append(nbrs.shape[1])

    # noise = torch.normal(mean=0, std=0.328, size = [16, nbrs.shape[1], 2])  # L1 0.1
    # # noise = 0.656  # L2 0.2
    # # noise = 0.984  # L3 0.3
    # # noise = 1.312  # L4 0.4
    # # noise = 1.640  # L5 0.5
    # # noise = 1.968  # L6 0.6
    # # noise = 2.297  # L7 0.7
    # # noise = 2.625  # L8 0.8
    # # noise = 2.953  # L9 0.9
    # # noise = 3.281  # L10 1

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

print(I)
print(NS)