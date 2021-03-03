
import warnings
warnings.filterwarnings("ignore")
import os
from tqdm import tqdm
import pandas as pd
import numpy as np
from scipy.signal import resample
from sklearn.model_selection import StratifiedKFold
from utils import  one_hot
np.random.seed(2020)
def jitter(x, snr_db):
    """
    根据信噪比添加噪声
    :param x:
    :param snr_db:
    :return:
    """
    # 随机选择信噪比
    assert isinstance(snr_db, list)
    snr_db_low = snr_db[0]
    snr_db_up = snr_db[1]
    snr_db = np.random.randint(snr_db_low, snr_db_up, (1,))[0]
    #print(snr_db)
    snr = 10 ** (snr_db / 10)
    Xp = np.sum(x ** 2, axis=0, keepdims=True) / x.shape[0]  # 计算信号功率
    Np = Xp / snr  # 计算噪声功率
    n = np.random.normal(size=x.shape, scale=np.sqrt(Np), loc=0.0)  # 计算噪声
    xn = x + n
    return xn

def cul_mod(X):
    return (X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2) ** .5

def DA_Permutation(X, nPerm=4, minSegLength=10):
    X_res = np.zeros(X.shape)
    for i in range(X.shape[0]):
        X_ = X[i, 0, :, :6]
        X_new = np.zeros((60, 6))
        idx = np.random.permutation(nPerm)
        bWhile = True
        while bWhile == True:
            segs = np.zeros(nPerm+1, dtype=int)
            segs[1:-1] = np.sort(np.random.randint(minSegLength, X_.shape[0]-minSegLength, nPerm-1))
            segs[-1] = X_.shape[0]
            if np.min(segs[1:]-segs[0:-1]) > minSegLength:
                bWhile = False
        pp = 0
        for ii in range(nPerm):
            x_temp = X_[segs[idx[ii]]:segs[idx[ii]+1],:]
            X_new[pp:pp+len(x_temp),:] = x_temp
            pp += len(x_temp)
        X_res[i, 0, :, :6] = X_new
        X_res[i, 0, :, 6] = cul_mod(X_new[:, :3])
        X_res[i, 0, :, 7] = cul_mod(X_new[:, 3:])
    return X_res

def standardization(X):
    # x1 = X.transpose(0, 1, 3, 2)
    x1 = X
    x2 = x1.reshape(-1, x1.shape[-1])
    mean = [8.03889039e-03, -6.41381949e-02, 2.37856977e-02, 8.64949391e-01,
            2.80964889e+00, 7.83041714e+00, 6.44853358e-01, 9.78580749e+00]
    std = [0.6120893, 0.53693888, 0.7116134, 3.22046385, 3.01195336, 2.61300056, 0.87194132, 0.68427254]
    mu=np.array(mean)
    sigma=np.array(std)
    x3 = ((x2 - mu) / (sigma))
    # x4 = x3.reshape(x1.shape).transpose(0, 1, 3, 2)
    x4 = x3.reshape(x1.shape)
    return x4


class XWDataset(object):
    def __init__(self,data_path,with_label=True,n_classes=19,**kwargs):

        self.data_path=data_path
        self.with_label=with_label #测试集无标签导入
        self.n_classes=n_classes
        #增加参数 with_nosie,
        self.with_nosie=kwargs.get("with_nosie",False)
        self.noise_SNR_db=kwargs.get("noise_SNR_db",[5,15])
        if self.with_nosie:
            print("添加随机噪声,SNR_db:{}".format(self.noise_SNR_db))
        self.load_dataset()

    @property
    def data(self):
        if self.with_label==True:
            return self.X,self.Y
        else:
            return self.X

    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, index):
        '''Generate one  of data'''

        x = self.X[int(index)]
        if self.with_label == True:
            y=self.Y[int(index)]
            y=one_hot(y,self.n_classes)
            return x,y
        else:
            return x
    @property
    def dim(self):
        return tuple(self.X.shape[1:])

    def load_dataset(self):
        df = pd.read_csv(self.data_path)
        # print(df.head())
        df = df.sort_values(['fragment_id', 'time_point'])
        ###特征提取
        df['mod'] = (df.acc_x ** 2 + df.acc_y ** 2 + df.acc_z ** 2) ** .5
        df['modg'] = (df.acc_xg ** 2 + df.acc_yg ** 2 + df.acc_zg ** 2) ** .5
        ###数据读取

        num = np.unique(df["fragment_id"]).shape[0]
        X_shape = (num,1, 60, 8)
        X = np.zeros(X_shape)
        for i in tqdm(range(X_shape[0])):
            tmp = df[df.fragment_id == i][:60]
            if self.with_label:
                arr = resample(tmp.drop(['fragment_id', 'time_point', 'behavior_id'],
                                        axis=1), 60, np.array(tmp.time_point))[0]
                X[i, 0, :, :] = arr
            else:
                arr = resample(tmp.drop(['fragment_id', 'time_point',],
                                        axis=1), 60, np.array(tmp.time_point))[0]
                X[i, 0, :, :] = arr
        ###############################################
        if self.with_label:
            #标准化
            X=standardization(X)
            Y = np.array(df.groupby("fragment_id")["behavior_id"].min())
            if self.with_nosie:
                print('noise')
                X1 = jitter(X, self.noise_SNR_db)
                X = np.concatenate([X, X1], axis=0)
                Y = np.concatenate([Y, Y], axis=0)
            self.X ,self.Y=X,Y
        else:
            # 标准化
            X = standardization(X)
            self.X=X
        self.fragment_ids = df.groupby("fragment_id")["fragment_id"].min()
        self.time_points = df.groupby("fragment_id")["time_point"]
        self.indexes = np.arange(self.X.shape[0])



    def stratifiedKFold(self,seed,fold=5):
        kfold = StratifiedKFold(fold, shuffle=True, random_state=seed)
        self.X_copy,self.Y_copy=self.X.copy(),self.Y.copy()
        self.train_valid_idxs=[ (train_idx,valid_idx) for train_idx,valid_idx in kfold.split(self.X_copy,self.Y_copy) ]

    def get_valid_data(self,index):
        """
        :param index:
        :return:  重新划分训练集和验证集 , 并返回验证集数据
        """
        train_idx,valid_idx= self.train_valid_idxs[index]
        X,Y= self.X_copy[train_idx],self.Y_copy[train_idx]
        nPerm = 4
        minSegLength = 2
        #print(X.shape)
        X_noise = jitter(X, [5, 15])    #添加噪音
        X_per = DA_Permutation(X,nPerm,minSegLength)    #置换
        X = np.concatenate([X, X_noise, X_per], axis=0)
        Y = np.concatenate([Y, Y, Y], axis=0)
        self.X, self.Y=X,Y
        self.valid_X,self.valid_Y=self.X_copy[valid_idx],self.Y_copy[valid_idx]
        return self.valid_X,self.valid_Y







