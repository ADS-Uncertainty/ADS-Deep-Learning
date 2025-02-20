import numpy as np
import pandas as pd

'''
Tips：
Case： threshold = np.percentile(uncertainty,95)    # Top 5% most uncertain samples
'''

for i in range(10):
    # Replace it with your file path
    # # MC Dropout
    # x_all = np.transpose(np.load('/media/TABLE-MCD/L_' +str(i+1) + '/L_' +str(i+1)+'_mux.npy'))
    # y_all = np.transpose(np.load('/media/TABLE-MCD/L_' +str(i+1) + '/L_' +str(i+1)+'_muy.npy'))
    # # Deep Ensembles
    x_all = np.transpose(np.load('/media/TABLE-DE/L_' +str(i+1) + '/L_' +str(i+1)+'_mux.npy'))
    y_all = np.transpose(np.load('/media/TABLE-DE/L_' +str(i+1) + '/L_' +str(i+1)+'_muy.npy'))

    gt_y = np.load('/media/TABLE-DE/L_0_y_gty.npy')
    re_gt_y = gt_y.reshape(-1,25)
    mask = np.any(re_gt_y == 0, axis=1)    # [False False False ...  True  True  True]
    # print(mask)
    # print(y_all)
    # print(y_all.shape)    # (10, 14315625)

    x_array = x_all[0].reshape(-1,25)[~mask].flatten()
    y_array = y_all[0].reshape(-1,25)[~mask].flatten()

    for x in range(9):
        re_x_all = x_all[x + 1].reshape(-1, 25)
        re_y_all = y_all[x + 1].reshape(-1, 25)
        new_column_x = re_x_all[~mask]
        new_column_y = re_y_all[~mask]
        result_x = new_column_x.flatten()
        result_y = new_column_y.flatten()
        x_array = np.vstack((x_array, result_x))
        y_array = np.vstack((y_array,result_y))

    x_std = np.std(x_array, axis=0, ddof=1).reshape(-1,25)
    y_std = np.std(y_array, axis=0, ddof=1).reshape(-1,25)
    x_sum = np.sum(x_std, axis=1)
    y_sum = np.sum(y_std, axis=1)

    all_sum = x_sum + y_sum

    # Threshold Setting
    # You need to change the threshold： Case-The number of samples * threshold = 80119
    Top_percent = np.argsort(all_sum)[::-1][:80119]    # 80119

    print('Done!')
    # np.save('/media/highuncertainty/NEW_Per15/MCD_L_' + str(i+1) + '.npy', Top_percent)
    np.save('/media/highuncertainty/NEW_Per15/DEs_L_' + str(i+1) + '.npy', Top_percent)