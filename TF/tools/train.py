import sys
import os
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import tensorflow as tf
from tools.tooltik import *
from model.ANN_PBLC import ANN
from model.GLM_PBLC import GLM
import os

gpus = tf.config.experimental.list_physical_devices('GPU')##自适应显存
tf.config.experimental.set_memory_growth(gpus[0], True)

if __name__ =='__main__':

    train_path = rootPath+'/data/train_data.csv'##############################写入训练地址
    test_path = rootPath+'/data/test_data.csv' ##############################写入测试地址

    train_db = read_data(train_path,512,100)   ##############路径/batch大小/乱序种子
    test_db = read_data(test_path,128,120)

    model = 'GLM' ##选择模型

    if model == 'ANN':
        save_reult = os.path.join(rootPath,'result/ANN')
        if os.path.exists(save_reult) == False:
            os.mkdir(save_reult)
        ANN(train_db,1,save_reult,test_db)
    elif model == 'GLM':
        save_reult = os.path.join(rootPath, 'result/GLM')
        if os.path.exists(save_reult) == False:
            os.mkdir(save_reult)
        GLM(train_db, 1, save_reult, test_db)

