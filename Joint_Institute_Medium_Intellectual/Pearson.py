import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import torch
import mpl_toolkits.axisartist as ast
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

Yf = []
Yp = []
length = 100*100 #1001*501
#fieldNames = ['T']
pccs_set = []
fieldNames = ['RHO', 'DIFF', 'VISC', 'T', 'SRC_PROG']
for iname in fieldNames:
    specy_fgm = pd.read_csv('01.orgData_orig/Y-'+iname+'.txt', header=None)
    specy_pre = pd.read_csv('preData1/'+iname+'.txt', header=None)
    for Z in range(100):
        for C in range(100):
            Yf.append(specy_fgm.at[Z, C])
            Yp.append(specy_pre.at[Z, C])
    Yff = np.array(Yf)
    Ypp = np.array(Yp)
    pccs = np.corrcoef(Yff, Ypp)
    #get the max accuracy as well as the min one
    print(iname + ": ", pccs[0][1])
    pccs_set.append(pccs[0][1])
    # if pccs[0][1] < 0.8:
    #     print(iname + ": ", pccs[0][1])
    #     print("The min value of the species", Yff.min(0))
    #     print("The max value of the species", Yff.max(0))
    Yf.clear()
    Yp.clear()
index_max = pccs_set.index(max(pccs_set))
print("The max accuracy of the chemicals is: " + fieldNames[index_max], max(pccs_set))
index_min = pccs_set.index(min(pccs_set))
print("The min accuracy of the chemicals is: " + fieldNames[index_min], min(pccs_set))
#画图
Z = 0
C = 0
counter = 0
Ymin = []
Ymax = []
the_Z = []
the_C = []
fhandZ = open('theZ.txt')
spcey_max = pd.read_csv('01.orgData_orig/Y-'+fieldNames[index_max]+'.txt', header=None)
spcey_min = pd.read_csv('01.orgData_orig/Y-'+fieldNames[index_min]+'.txt', header=None)
for lineZ in fhandZ:
    fhandC = open('theC.txt')
    for lineC in fhandC:
        the_Z.append(float(lineZ)) #??
        the_C.append(float(lineC))
        Ymax.append(spcey_max.at[Z, C])
        Ymin.append(spcey_min.at[Z, C])
        C += 1
        counter += 1
    C = 0
    Z += 1
the_Z = np.array(the_Z)
the_C = np.array(the_C)
Ymin = np.array(Ymin)
Ymax = np.array(Ymax)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(the_Z, the_C, Ymax, c='b', marker='o')
plt.xlabel('Z value')
plt.ylabel('C value')
plt.show()




