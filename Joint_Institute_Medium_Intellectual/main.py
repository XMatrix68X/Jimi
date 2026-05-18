import os
import argparse
import torch
import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import torch.nn.functional as F
import netTest
import matplotlib.pyplot as plt
#ARsrc全是0不参与， HEAVIBETA全是1不参与, ENTH全是0不参与（查出原表中第几个）
graphsPATH = '02.annGraph/graphs/'
fieldNames = ['RHO', 'DIFF', 'VISC', 'T', 'SRC_PROG']
#the second
#fieldNames = ['CO2', 'O2', 'CO', 'CH4', 'H2O', 'H2', 'OH', 'H', 'O', 'HO2', 'H2O2', 'C', 'CH', 'CH2', 'CH2(S)', 'CH3', 'HCO', 'CH2O', 'CH2OH', 'CH3O', 'CH3OH', 'C2H', 'C2H2', 'C2H3', 'C2H4', 'C2H5', 'C2H6', 'HCCO', 'CH2CO', 'HCCOH', 'N', 'NH', 'NH2', 'NH3', 'NNH', 'NO', 'NO2', 'N2O', 'HNO', 'CN', 'HCN', 'H2CN', 'HCNN', 'HCNO', 'HOCN', 'HNCO', 'NCO', 'N2', 'AR', 'C3H7', 'C3H8', 'CH2CHO', 'CH3CHO']
#fieldNames = ['RHO', 'DIFF', 'VISC', 'CP', 'T', 'CO2', 'O2', 'CO', 'CH4', 'H2O', 'H2', 'OH', 'H', 'O', 'HO2', 'H2O2', 'C', 'CH', 'CH2', 'CH2(S)', 'CH3', 'HCO', 'CH2O', 'CH2OH', 'CH3O', 'CH3OH', 'C2H', 'C2H2', 'C2H3', 'C2H4', 'C2H5', 'C2H6', 'HCCO', 'CH2CO', 'HCCOH', 'N', 'NH', 'NH2', 'NH3', 'NNH', 'NO', 'NO2', 'N2O', 'HNO', 'CN', 'HCN', 'H2CN', 'HCNN', 'HCNO', 'HOCN', 'HNCO', 'NCO', 'N2', 'AR', 'C3H7', 'C3H8', 'CH2CHO', 'CH3CHO']#, 'CO2Src', 'O2Src', 'COSrc', 'CH4Src', 'H2OSrc', 'H2Src', 'OHSrc', 'HSrc', 'OSrc', 'HO2Src', 'H2O2Src', 'CSrc', 'CHSrc', 'CH2Src', 'CH2(S)Src', 'CH3Src', 'HCOSrc', 'CH2OSrc', 'CH2OHSrc', 'CH3OSrc', 'CH3OHSrc', 'C2HSrc', 'C2H2Src', 'C2H3Src', 'C2H4Src', 'C2H5Src', 'C2H6Src', 'HCCOSrc', 'CH2COSrc', 'HCCOHSrc', 'NSrc', 'NHSrc', 'NH2Src', 'NH3Src', 'NNHSrc', 'NOSrc', 'NO2Src', 'N2OSrc', 'HNOSrc', 'CNSrc', 'HCNSrc', 'H2CNSrc', 'HCNNSrc', 'HCNOSrc', 'HOCNSrc', 'HNCOSrc', 'NCOSrc', 'N2Src', 'C3H7Src', 'C3H8Src', 'CH2CHOSrc', 'CH3CHOSrc', 'SRC_ZMIX', 'Z2RHODOT', 'ZSRC_ZMIX', 'SRC_PROG', 'PROG', 'SRC_RAD']
# ,'C7H14OOH','CH3CHO','H2O','NC7KET','C2H3','C3H5-A','C7H14O','CH3COCH2','H2','O2','C2H4','C3H5O','C7H15-1','CH3CO','HCCO','OH','C2H5CHO','C3H6','C7H15O2','CH3O2H','HCO','O','C2H5COCH2','C4H6','C7H15','CH3O2','HO2','PC4H9','C2H5O','C4H7O','C7H16','CH3OH','H','C2H5','C4H7','CH2CHO','CH3O','N2','C2H6','C4H8-1','CH2CO','CH3','NC3H7CHO','C2H','C5H10-1','CH2OH','CH4','NC3H7COCH2','C2H2','C3H2','C5H11-1','CH2O','CO2','NC3H7COCH3','C2H3CHO','C3H3','C5H9','CH2-S','CO','NC3H7CO','C2H3CO','C3H4-A','C7H14OOHO2','CH3CHCO','H2O2','NC3H7']


# 训练次数
TRAIN_TIMES = 300000#2040000#68*30000
PATIENCE = 100
PRIDICTION_TOLERANCE = 0.001#1%TOLERANCE of the real data
# 输入输出的数据维度，这里都是1维
INPUT_FEATURE_DIM = 2
OUTPUT_FEATURE_DIM = len(fieldNames)
# 隐含层中神经元的个数
# 8,16,32->10,20,40
HIDDEN_1_DIM = 10
HIDDEN_2_DIM = 20
HIDDEN_3_DIM = 40
HIDDEN_4_DIM = 20
HIDDEN_5_DIM = 10
#initial 0.001
LEARNING_RATE = 0.0001
print(OUTPUT_FEATURE_DIM)



NFGM = []# list of dataframes
for iname in fieldNames:
    speciesI = pd.read_csv('01.orgData_LB/'+iname+'.txt', header=None)#这里是标签存放的地址
    NFGM.append(speciesI)

#the following is used for debug. By WZ
#for i, iFGM in enumerate(NFGM):
#    print(iFGM.at[1,0])
#    exit()

# ============================ step 1/6 导入数据 ============================
parser = argparse.ArgumentParser(description='give a filed name and learning rate, for example T')
parser.add_argument('--learningRate', type=float, default = LEARNING_RATE)
args = parser.parse_args()
print(args.learningRate)
LEARNING_RATE = args.learningRate


# Create directory
if not os.path.exists(graphsPATH):
    os.mkdir(graphsPATH)
    print("Directory " , graphsPATH ,  " Created ")
else:
    print("Directory " , graphsPATH ,  " already exists")




# 数据构造
# 这里x_data、y_data都是tensor格式，在PyTorch0.4版本以后，也能进行反向传播
# 所以不需要再转成Variable格式了
# linspace函数用于生成一系列数据
# unsqueeze函数可以将一维数据变成二维数据，在torch中只能处理二维数据
# The number of Z and PV for creating the input array，(this part need to be modified) by WZ
Ztarget = []
Ctarget = []
fhandZ = open('theZ_B.txt')#大表就是_B
for line in fhandZ:
    Ztarget.append(float(line))
fhandC = open('theC_B.txt')
for line in fhandC:
    Ctarget.append(float(line))




x=Ztarget
y=Ctarget

inputs_np = np.zeros((len(x)*len(y),INPUT_FEATURE_DIM ))
outputs_np = np.zeros((len(x)*len(y),OUTPUT_FEATURE_DIM ))
index = -1

#数据标签的预处理
for Z in range(len(x)):
    for C in range(len(y)):
        index += 1
        inputs_np[index] = np.array([x[Z],y[C]])
        for i, iFGM in enumerate(NFGM):
            outputs_np[index][i] = iFGM.at[Z,C]#这是建立了个二维表，x和y轴分别为C和Z

#64就把.float换成.double
x_data = torch.from_numpy(inputs_np).float()
y_data = torch.from_numpy(outputs_np).float()


#下面是对标签的归一化处理
y_data_real = y_data
Min = y_data_real.min(0).values
Max = y_data_real.max(0).values
#获得数据进行还原
# print(Min)
# print(Max)
# exit()
y_data = (y_data_real-Min)/(Max-Min)

# ============================ step 2/6 选择模型 ============================

# 建立网络，初次训练
net = netTest.Batch_Net_5_2(INPUT_FEATURE_DIM, HIDDEN_1_DIM, HIDDEN_2_DIM, HIDDEN_3_DIM, HIDDEN_4_DIM, HIDDEN_5_DIM, OUTPUT_FEATURE_DIM)
#加载网络，在已有的基础上继续
net.load_state_dict(torch.load('Target_network.pkl'))
# init_params(net)
print(net)
Stored_net = net

# ============================ step 3/6 选择优化器   =========================

optimizer = torch.optim.Adam(net.parameters(), lr=LEARNING_RATE)

# ============================ step 4/6 选择损失函数 =========================
# 定义一个目标函数（损失函数）
# Mean Square Error (MSE), MSELoss(), L2 loss
# Mean Absolute Error (MAE), MAELoss(), L1 Loss
loss_func = torch.nn.MSELoss()
INITIAL_LOSS = loss_func(0*y_data, y_data)
LOSS_TOLERANCE = loss_func(0*y_data, PRIDICTION_TOLERANCE*y_data).data.numpy()
print('LOSS_TOLERANCE',LOSS_TOLERANCE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.5, threshold=PATIENCE/TRAIN_TIMES*LOSS_TOLERANCE, patience=PATIENCE)

# ============================ step 5/6 模型训练 ============================
LOSS_min = 0
for i in range(TRAIN_TIMES):
    # 输入数据进行预测
    prediction = net(x_data)
    #prediction_real = prediction*(Max-Min)+Min
    # 计算预测值与真值误差，注意参数顺序问题
    # 第一个参数为预测值，第二个为真值
    loss = loss_func(prediction, y_data)#*100/firstLoss
    # Change the learning rate
    #scheduler.step()
    scheduler.step(loss)
    # 开始优化步骤
    # 每次开始优化前将梯度置为0
    optimizer.zero_grad()
    # 误差反向传播
    loss.backward()
    optimizer.step()
    # 按照最小loss优化参数
    # 可视化训练结果
    # 需要算一下accuracy了
    LOSS_SQRT = np.sqrt(loss.data.numpy() / INITIAL_LOSS.data.numpy())
    print("Iteration : ",'%05d'%i, "\tLearningRate : {:.3e}\tLoss: {:.4e}\tRelativeError:{:.5e}".format( optimizer.param_groups[0]['lr'], loss.data.numpy(), LOSS_SQRT))
    # 获取loss最小的model
    if i == 0:
        LOSS_min = LOSS_SQRT
        Stored_net = net
    if LOSS_SQRT < LOSS_min:
        Stored_net = net
        LOSS_min = LOSS_SQRT
    #每1000次保存一次
    if i % 1000 == 0:
        traced_script_module = torch.jit.trace(Stored_net, torch.rand(INPUT_FEATURE_DIM))
        traced_script_module.save('Target_network.pt')
        torch.save(Stored_net.state_dict(), 'Target_network.pkl')
    if LOSS_SQRT < PRIDICTION_TOLERANCE:
        break    # Lower than tollance break here
    else:
        continue

# ============================ step 6/6 保存模型 ============================
traced_script_module = torch.jit.trace(Stored_net, torch.rand(INPUT_FEATURE_DIM))
print(LOSS_min)
traced_script_module.save('Target_network.pt')
torch.save(Stored_net.state_dict(), 'Target_network.pkl')



