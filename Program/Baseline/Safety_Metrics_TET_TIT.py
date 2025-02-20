import pandas as pd
import numpy as np
import os
import pickle
pd.set_option('display.width', None)

'''
Tips：
The calculation data comes from /Data Production/3_Safety_data_production.py
'''

# Replace it with your data file path
# NOISE_LEVEL = [0.328,0.656,0.984,1.312,1.64,1.968,2.297,2.625,2.953,3.281]
predict_data_y = np.load('/media/safetydatacollect/3.281_muy.npy')
# load data (Waiting for upload)
df = np.load('/media/safetydatacollect/ENDPREDATASUPPORT_dele0.npy')

# predict_data_x = np.transpose(predict_data_x)
predict_data_y = np.transpose(predict_data_y)

rows2 = df.shape[1]
Wrong_one = []
for ii in range(0, rows2, 25):
    Wrong_one.append(ii)

FINAL_TET = []
FINAL_TIT = []
print('begin!')

for i in range(10):
    # load data (Waiting for upload)
    end = np.load('/media/safetydatacollect/ENDPREDATASUPPORT.npy')
    # print(end.shape)     # (26, 14315625)

    point_array = end[25]
    # point = np.where(point_array < 0)    # ok!
    # print(len(point[0]))

    num = end.shape[1] // 25  # 572625
    data_reshaped = point_array.reshape(num, 25)

    mask = np.any(data_reshaped < 0, axis=1)
    print(mask.shape)

    rhp1 = predict_data_y[i].reshape(num, 25)
    dd1 = rhp1[~mask]
    resulty = dd1.flatten()

    # calculate dl_v --y:
    temple1 = np.insert(resulty, 0, 0)
    temple2 = temple1[:-1]
    dl_v = (resulty - temple2) / 0.2

    # check!
    array_distance = df[25] - resulty - df[18]    # df['distance'] = df['re_pre_y'] - df['pred_muy'] - df['pre_v_Length']
    ind = np.where(array_distance <= 0)
    print('Crash:', len(ind[0]))

    array_dis_v = dl_v - df[22]                   # df['dis_v'] = df['dl_v'] - df['pre_v_Vel']
    ind_dis_v = np.where(array_dis_v <= 0)
    print('Safe_v:', len(ind_dis_v[0]))

    useless_columns = set(Wrong_one + list(ind[0]) + list(ind_dis_v[0]))
    useless_columns = sorted(useless_columns)

    all_columns = set(list(range(0, rows2)))
    use_columns = list(all_columns.difference(useless_columns))

    # TTC = all_data
    all_data = array_distance / array_dis_v  # df['re_pre_y'] - df['pred_muy'] - df['pre_v_Length']) / (df['dl_v'] - df['pre_v_Vel']
    # all_data = [round(num, 3) for num in array_TTC]

    mask = np.zeros_like(all_data, dtype=bool)
    mask[useless_columns] = True
    all_data[mask] = -1

    TET_DATA = []
    TIT_DATA = []
    ttc_threshold = 3

    for da in range(0,rows2,25):
        data = np.array(all_data[da:da+25])
        if np.any(data > 0):
            # target = sum(filter(lambda x: x > 0, data))
            # TTC_DATA.append(target)

            # TET
            tet_tar = sum(1 for Y in data if 0<Y<ttc_threshold) * 0.2
            # TIT
            TET_DATA.append(tet_tar)

            tit_tar = [x for x in data if 0<x<ttc_threshold]
            if tit_tar != []:
                temp = []
                for ti in tit_tar:
                    target_tit = (ttc_threshold - ti) * 0.2
                    temp.append(target_tit)
                TIT_DATA.append(sum(temp))

    Average_TET = sum(TET_DATA) / len(TET_DATA)
    Average_TIT = sum(TIT_DATA) / len(TET_DATA)

    # print('Useful TTC traj:', len(TTC_DATA), '| Average TTC:', Average_TTC)
    print('===================================================================')
    print('Useful TET traj:', len(TET_DATA), '| Average TET:', Average_TET)
    print('Useful TET traj:', len(TET_DATA), '| Average TIT:', Average_TIT)

    FINAL_TET.append(Average_TET)
    FINAL_TIT.append(Average_TIT)

print('FINAL_TET:', FINAL_TET)
print('FINAL_TIT:', FINAL_TIT)

mean_tet = np.mean(FINAL_TET)
std_tet = np.std(FINAL_TET,axis=0,ddof=1)

mean_tit = np.mean(FINAL_TIT)
std_tit = np.std(FINAL_TIT,axis=0,ddof=1)
print('TET:',mean_tet,'±',std_tet)
print('TIT:',mean_tit,'±',std_tit)