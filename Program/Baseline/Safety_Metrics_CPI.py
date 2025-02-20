import pandas as pd
import numpy as np
pd.set_option('display.width', None)

'''
Tips：
The calculation data comes from /Data Production/3_Safety_data_production.py
'''

# Replace it with your data file path
# load data (Waiting for upload)
df = np.load('/media/simplexity/d7591000-e74c-4eb0-a7f9-3b3780ea1e09/wyf/Record/safetydatacollect/ENDPREDATASUPPORT_dele0.npy')
print(df.shape)

NOISE_LEVEL = [0.328,0.656,0.984,1.312,1.64,1.968,2.297,2.625,2.953,3.281]

for aa in NOISE_LEVEL:
    # Replace it with your data file path
    predict_data_y = np.load('/media/safetydatacollect/' + str(aa) + '_muy.npy')
    print(str(aa))
    # predict_data_x = np.transpose(predict_data_x)
    predict_data_y = np.transpose(predict_data_y)
    rows2 = df.shape[1]

    FINAL_CPI = []
    print('begin!')

    for i in range(10):
        # Replace it with your data file path
        # load data (Waiting for upload)
        end = np.load('/media/safetydatacollect/ENDPREDATASUPPORT.npy')

        point_array = end[25]

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
        # array_distance = df[25] - (df[5]-df[21]+df[25]) - df[18]    # df['distance'] = df['re_pre_y'] - df['pred_muy'] - df['pre_v_Length']
        array_distance = df[25] - resulty - df[18]    # df['distance'] = df['re_pre_y'] - df['pred_muy'] - df['pre_v_Length']

        ind = np.where(array_distance <= 0)
        print('Crash:', len(ind[0]))

        array_dis_v = dl_v - df[22]                   # df['dis_v'] = df['dl_v'] - df['pre_v_Vel']
        ind_dis_v = np.where(array_dis_v <= 0)
        print('Safe_v:', len(ind_dis_v[0]))

        Wrong_one = []
        for ii in range(0, rows2, 25):
            Wrong_one.append(ii)

        useless_columns = set(Wrong_one + list(ind[0]) + list(ind_dis_v[0]))
        useless_columns = sorted(useless_columns)

        all_columns = set(list(range(0, rows2)))
        use_columns = list(all_columns.difference(useless_columns))

        all_data = (array_dis_v ** 2) / (2 * array_distance)
        mask = np.zeros_like(all_data, dtype=bool)
        mask[useless_columns] = True
        all_data[mask] = -1

        CPI_DATA = []
        ttc_threshold = 3

        for da in range(0, rows2, 25):
            data = np.array(all_data[da:da + 25])
            if np.any(data > 0):
                # aaa = list(filter(lambda x:  x > 0, data))
                A_tar = sum(1 for Y in data if 27.707 < Y)
                B_tar = sum(1 for Y in data if 0 < Y)
                # if len(aaa) != 0 :
                target = A_tar/B_tar
                CPI_DATA.append(target)
        # print(DRAC_DATA)
        Average_CPI = sum(CPI_DATA) / len(CPI_DATA)
        print('Useful DRAC traj:', len(CPI_DATA), '| Average DRAC:', Average_CPI)

        FINAL_CPI.append(Average_CPI)

    print('FINAL_CPI:', FINAL_CPI)
    mean_cpi = np.mean(FINAL_CPI)
    std_cpi = np.std(FINAL_CPI,axis=0,ddof=1)
    print('CPI:',mean_cpi,'±',std_cpi)