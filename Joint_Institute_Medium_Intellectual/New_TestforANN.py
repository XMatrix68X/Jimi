# !/usr/bin/python
# coding: utf8
# @Time    : 2020-03-25 19:09

import os
import argparse
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
import netTest

#fieldNames = ['T']
fieldNames = ['RHO', 'DIFF', 'VISC', 'T', 'SRC_PROG']
#fieldNames = ['CO2', 'O2', 'CO', 'CH4', 'H2O', 'H2', 'OH', 'H', 'O', 'HO2', 'H2O2', 'C', 'CH', 'CH2', 'CH2(S)', 'CH3', 'HCO', 'CH2O', 'CH2OH', 'CH3O', 'CH3OH', 'C2H', 'C2H2', 'C2H3', 'C2H4', 'C2H5', 'C2H6', 'HCCO', 'CH2CO', 'HCCOH', 'N', 'NH', 'NH2', 'NH3', 'NNH', 'NO', 'NO2', 'N2O', 'HNO', 'CN', 'HCN', 'H2CN', 'HCNN', 'HCNO', 'HOCN', 'HNCO', 'NCO', 'N2', 'AR', 'C3H7', 'C3H8', 'CH2CHO', 'CH3CHO']
# ,'C7H14OOH','CH3CHO','H2O','NC7KET','C2H3','C3H5-A','C7H14O','CH3COCH2','H2','O2','C2H4','C3H5O','C7H15-1','CH3CO','HCCO','OH','C2H5CHO','C3H6','C7H15O2','CH3O2H','HCO','O','C2H5COCH2','C4H6','C7H15','CH3O2','HO2','PC4H9','C2H5O','C4H7O','C7H16','CH3OH','H','C2H5','C4H7','CH2CHO','CH3O','N2','C2H6','C4H8-1','CH2CO','CH3','NC3H7CHO','C2H','C5H10-1','CH2OH','CH4','NC3H7COCH2','C2H2','C3H2','C5H11-1','CH2O','CO2','NC3H7COCH3','C2H3CHO','C3H3','C5H9','CH2-S','CO','NC3H7CO','C2H3CO','C3H4-A','C7H14OOHO2','CH3CHCO','H2O2','NC3H7']
# 输入输出的数据维度，这里都是1维
INPUT_FEATURE_DIM = 2
OUTPUT_FEATURE_DIM = len(fieldNames)
# 隐含层中神经元的个数
field_num = 0
Min_data = np.array([1.4700e-01,  1.4657e-05,  1.1460e-05,  3.0000e+02, -3.1900e-04])
Max_data = np.array([1.1569e+00, 9.5360e-05, 7.1730e-05, 2.2268e+03, 9.6972e+03])
# Min_data = np.array([1.4699e-01, 1.4657e-05, 1.1460e-05, 1.0099e+03, 3.0000e+02, 0.0000e+00,
#         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00,
#         0.0000e+00, 0.0000e+00, 0.0000e+00, 0.0000e+00])
# Max_data = np.array([1.1569e+00, 9.5360e-05, 7.1731e-05, 2.7483e+03, 2.2268e+03, 1.3698e-01,
#         2.3300e-01, 1.1004e-01, 1.0000e+00, 1.2808e-01, 1.2257e-02, 4.7519e-03,
#         2.7053e-04, 2.5488e-03, 1.4707e-04, 1.6724e-03, 5.5428e-06, 6.0652e-06,
#         4.2235e-05, 3.5941e-06, 1.9454e-03, 4.0362e-05, 5.3369e-03, 8.3197e-06,
#         3.0166e-06, 4.3416e-03, 2.2052e-06, 1.4586e-02, 2.7430e-05, 4.0253e-03,
#         3.4433e-05, 2.5334e-03, 3.7664e-05, 6.1687e-04, 1.8797e-05, 8.1530e-07,
#         3.3943e-07, 9.5543e-07, 8.8931e-05, 1.1755e-08, 1.4183e-03, 1.2268e-05,
#         8.2871e-07, 1.5584e-07, 1.7309e-07, 1.9144e-04, 2.0688e-08, 1.6286e-08,
#         4.9203e-06, 1.9021e-07, 2.0867e-05, 6.2947e-07, 7.6700e-01, 1.2068e-20,
#         8.5166e-08, 3.9902e-05, 1.8432e-06, 3.6047e-05])
Min_data = torch.from_numpy(Min_data)
Max_data = torch.from_numpy(Max_data)
HIDDEN_1_DIM = 10
HIDDEN_2_DIM = 20
HIDDEN_3_DIM = 40
HIDDEN_4_DIM = 20
HIDDEN_5_DIM = 10
LEARNING_RATE = 0.001
net = netTest.Batch_Net_5_2(INPUT_FEATURE_DIM, HIDDEN_1_DIM, HIDDEN_2_DIM, HIDDEN_3_DIM, HIDDEN_4_DIM, HIDDEN_5_DIM, OUTPUT_FEATURE_DIM)
model = net.load_state_dict(torch.load('Target_network.pkl'))
fz_handle = open('theZ.txt')
fc_handle = open('theC.txt')
Z = []
C = []
for line in fz_handle:
    Z.append(float(line))
for line in fc_handle:
    C.append(float(line))
for index_z in range(len(Z)):
    for index_c in range(len(C)):
        test_in = np.array([Z[index_z], C[index_c]])
        x_data = torch.from_numpy(test_in).float()
        prediction = net(x_data) * (Max_data - Min_data) + Min_data
        for iname in fieldNames:
            f_handle = open('preData1/' + iname + '.txt', 'a')
            f_handle.write(str(float(prediction[field_num])) + ",")
            #判断一下是否要换行
            if index_c == len(C) - 1:
                f_handle.write('\n')
            f_handle.close()
            field_num += 1
        field_num = 0
fz_handle.close()
fc_handle.close()
