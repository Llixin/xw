import os
import shutil
import numpy as np
import pandas as pd
from tqdm import tqdm
from scipy.signal import resample
from Model.Model_CRNN import CRNN
from Model.Model_CNN import CNN
from Model.Model_Res2Net import Res2Net, CELoss
from keras.optimizers import Adam as keras_adam
from keras.utils import to_categorical
import keras.backend as K
from sklearn.model_selection import StratifiedKFold
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from Model.Sub_block import zero_loss
from Model.mixup_generator import MixupGenerator_center
from dataset import  XWDataset
from metrics import  XWMetrics
from torch.utils.data import DataLoader
from utils import  one_hot
from torch_func import  Agent
import gc
from numba import cuda

from torch.optim import lr_scheduler
from torch.optim import Adam as torch_adam
import torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE_INFO=dict(
    gpu_num=torch.cuda.device_count(),
    device_ids = range(0, torch.cuda.device_count(), 1),
    device = torch.device("cuda:0") if torch.cuda.is_available() else "cpu",
    index_cuda=0, )

if not os.path.exists('./save_dir'):
    os.makedirs('./save_dir')

def acc_combo(y, y_pred):
    # print(y)
    # print(y_pred)
    # 数值ID与行为编码的对应关系
    mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
        4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5',
        8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
        12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
        16: 'C_2', 17: 'C_5', 18: 'C_6'}
    # 将行为ID转为编码
    code_y, code_y_pred = mapping[y], mapping[y_pred]
    if code_y == code_y_pred: #编码完全相同得分1.0
        return 1.0
    elif code_y.split("_")[0] == code_y_pred.split("_")[0]: #编码仅字母部分相同得分1.0/7
        return 1.0/7
    elif code_y.split("_")[1] == code_y_pred.split("_")[1]: #编码仅数字部分相同得分1.0/3
        return 1.0/3
    else:
        return 0.0


def cul_acc_combo(y, y_pred):
    sample_num = len(y)
    score_ = []
    for i in range(sample_num):
        score_.append(acc_combo(y[i],y_pred[i]))
    return np.mean(score_)



def DA_Permutation(X, nPerm=4, minSegLength=10):
    # data augmentation: permutation
    def cul_mod(X):
        # calculate mod of acceleration
        return (X[:, 0] ** 2 + X[:, 1] ** 2 + X[:, 2] ** 2) ** .5

    X_res = np.zeros(X.shape)
    for i in range(X.shape[0]):
        X_ = X[i, :, :6, 0]
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
        X_res[i, :, :6, 0] = X_new
        X_res[i, :, 6, 0] = cul_mod(X_new[:, :3])
        X_res[i, :, 7, 0] = cul_mod(X_new[:, 3:])
    return X_res

def load_dataset():
    # read data
    train = pd.read_csv('./data/sensor_train.csv')
    test = pd.read_csv('./data/sensor_test.csv')
    sub = pd.read_csv('./data/提交结果示例.csv')
    y = train.groupby('fragment_id')['behavior_id'].min()
    y= np.array(y)
    train['mod'] = (train.acc_x ** 2 + train.acc_y ** 2 + train.acc_z ** 2) ** .5
    train['modg'] = (train.acc_xg ** 2 + train.acc_yg ** 2 + train.acc_zg ** 2) ** .5
    test['mod'] = (test.acc_x ** 2 + test.acc_y ** 2 + test.acc_z ** 2) ** .5
    test['modg'] = (test.acc_xg ** 2 + test.acc_yg ** 2 + test.acc_zg ** 2) ** .5

    x = np.zeros((7292, 60, 8, 1))
    t = np.zeros((7500, 60, 8, 1))
    
    for i in tqdm(range(7292)):
        tmp = train[train.fragment_id == i][:60]
        x[i,:,:, 0] = resample(tmp.drop(['fragment_id', 'time_point', 'behavior_id'],
                                        axis=1), 60, np.array(tmp.time_point))[0]
    for i in tqdm(range(7500)):
        tmp = test[test.fragment_id == i][:60]
        t[i,:,:, 0] = resample(tmp.drop(['fragment_id', 'time_point'],
                                        axis=1), 60, np.array(tmp.time_point))[0]
    return x, y, t, sub

def inference_with_keras(Net, dataset, save_dir):
    """
    Inference CNN or CRNN with keras
    Not usiong Normalization
    Data augmentation: mixup, permutation
    Loss: categorical_crossentropy and center_loss
    """
    x, y, t, sub = dataset
    lambda_centerloss=0.005
    batch_size = 256

    valid_pred_list = []
    proba_t = np.zeros((7500, 19))
    nPerm = 4
    minSegLength = 2
    epochs = 1000
    seeds = [1]
    learning_rate = 3e-4
    count = 0
    t_dummy = np.zeros((7500, 19))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for seed in seeds:
        save_model_dir=os.path.join(save_dir, 'seed'+str(seed))
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        kfold = StratifiedKFold(10, shuffle=True, random_state=seed)
        for fold, (xx, yy) in enumerate(kfold.split(x, y)):
            x_train = x[xx]
            x_val = x[yy]
            x_per = DA_Permutation(x_train, nPerm=nPerm, minSegLength=minSegLength)
            x_train = np.concatenate([x_train, x_per], axis=0)
            y_ = to_categorical(y, num_classes=19)
            y_train = np.concatenate([y_[xx], y_[xx]], axis=0)
            y_val = y_[yy]
            dummy1 = np.zeros((x_train.shape[0], 1))
            dummy2 = np.zeros((x_val.shape[0], 1))

            model = Net()
            # model.compile(loss=['categorical_crossentropy', zero_loss],
            #               loss_weights=[1, lambda_centerloss],
            #
            #               optimizer=keras_adam(learning_rate),
            #               metrics=['acc'])
            model.summary()

            # early_stopping = EarlyStopping(monitor='val_behaviour_acc',
            #                                verbose=0,
            #                                mode='max',
            #                                patience=200)
            # checkpoint = ModelCheckpoint(save_model_dir+'/'+f'model{fold}.h5',
            #                              monitor='val_behaviour_acc',
            #                              verbose=1,
            #                              mode='max',
            #                              save_best_only=True)
            #
            # training_generator = MixupGenerator_center(x_train, y_train, dummy1, batch_size=batch_size, alpha=0.3)()

            # model.fit_generator(generator=training_generator,
            #                 steps_per_epoch=x_train.shape[0] // batch_size,
            #           epochs=epochs,
            #           verbose=1,
            #           shuffle=True,
            #           validation_data=([x_val, y_val], [y_val, dummy2]),
            #           callbacks=[early_stopping, checkpoint])
            print('model'+str(count))
            model.load_weights(save_model_dir+'/'+f'model{fold}.h5')
            p, _ = model.predict([t, t_dummy], verbose=0, batch_size=batch_size)

            proba_t += p
            valid_pred, _ = model.predict([x_val, y_val], verbose=0, batch_size=batch_size)
            valid_pred = np.argmax(valid_pred, axis=1)
            acc = cul_acc_combo(y[yy], valid_pred)
            print(acc)
            valid_pred_list.append(acc)
            count += 1


    print('acc combo of val set: ')
    print(valid_pred_list)
    proba_t = proba_t / count
    valid_acc_mean = sum(valid_pred_list) / count
    print('mean of acc combo: '+str(valid_acc_mean))
    sub.behavior_id = np.argmax(proba_t, axis=1)
    np.save(os.path.join(save_dir, "proba.npy"), proba_t)
    sub.to_csv(os.path.join(save_dir, str(round(valid_acc_mean, 3))) + '_sub.csv', index=False)


def fusion_model(save_dirs, weight=None):
    """
    对多模型预测的结果进行加权平均
    """
    proba = np.zeros((7500, 19))
    if weight is None:
        for i, save_dir in enumerate(save_dirs):
            p = np.load(save_dir+'/proba.npy')
            proba = proba + p
        proba = proba / len(save_dirs)
    else:
        for i, save_dir in enumerate(save_dirs):
            p = np.load(save_dir+'/proba.npy')
            proba = proba + p * weight[i]
    return proba

def inference_with_pytorch(save_dir):
    """
    Inference Res2Net with pytorch
    Using Normalization
    Data augmentation: jitter, permutation
    Loss: categorical_crossentropy
    """
    sub = pd.read_csv('./data/提交结果示例.csv')
    NUM_CLASSES = 19
    METRICS=XWMetrics()
    train_data=XWDataset("./data/sensor_train.csv",with_label=True)
    test_data=XWDataset("./data/sensor_test.csv",with_label=False)
    proba_t = np.zeros((len(test_data), NUM_CLASSES))
    folds=10
    seeds = [1]
    valid_acc_list = []
    count = 0
    EPOCH = 100
    BATCH_SIZE = 256
    for seed in seeds:
        save_model_dir=os.path.join(save_dir, 'seed'+str(seed))
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        train_data.stratifiedKFold(seed, folds)
        for fold in range(folds):
            model=Res2Net(num_classes=NUM_CLASSES)
            save_model_fold_dir=os.path.join(save_model_dir,"fold_{}".format(fold))
            agent=Agent(model=model,device_info=DEVICE_INFO, save_dir=save_model_fold_dir)
            # earlyStopping = None

            # LOSS={ "celoss":(CELoss(), 1.0)}
            #LOSS_WEIGHT = {"celoss": 1.0, "centerloss": 0.005}
            # OPTIM=torch_adam(model.parameters(), lr=3e-4)
            # reduceLR = lr_scheduler.ReduceLROnPlateau(OPTIM, mode="max", factor=0.5, patience=8, verbose=True, min_lr=1e-5)
            # agent.compile(loss_dict=LOSS,optimizer=OPTIM, metrics=METRICS)
            agent.summary()
            valid_X,valid_Y=train_data.get_valid_data(fold)
            valid_Y=one_hot(valid_Y,NUM_CLASSES)
            valid_data = [(valid_X[i],valid_Y[i]) for i in range(valid_X.shape[0])]

            # train_generator=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
            # agent.fit_generator(train_generator, epochs=EPOCH,
            #                               validation_data=valid_data,
            #                         reduceLR=reduceLR,
            #                         earlyStopping=earlyStopping)

            agent.load_best_model()
            valid_pred = agent.predict(valid_X, batch_size=BATCH_SIZE,phase="valid")
            print(valid_pred.shape)
            acc = cul_acc_combo(np.argmax(valid_Y, axis=1), np.argmax(valid_pred, axis=1))
            print(acc)
            valid_acc_list.append(acc)

            test_X=[test_data.data[i] for i in range(test_data.data.shape[0])]
            scores_test= agent.predict(test_X,batch_size=BATCH_SIZE,phase="test")
            proba_t+=scores_test
            count += 1
    print('acc combo of val set: ')
    print(valid_acc_list)
    proba_t = proba_t / count
    valid_acc_mean = sum(valid_acc_list) / count
    print('mean of acc combo: '+str(valid_acc_mean))
    sub.behavior_id = np.argmax(proba_t, axis=1)
    np.save(os.path.join(save_dir, "proba.npy"), proba_t)
    sub.to_csv(os.path.join(save_dir, str(round(valid_acc_mean, 3))) + '_sub.csv', index=False)

def main():
    dataset = load_dataset()    # x, y, t, sub
    sub = dataset[3]
    cnn_save_dir = "./final_model/cnn"
    crnn_save_dir = "./final_model/crnn"
    res2net_save_dir = "./final_model/res2net"

    #train pure cnn
    inference_with_keras(CNN, dataset, cnn_save_dir)

    #train crnn
    inference_with_keras(CRNN, dataset, crnn_save_dir)

    #train res2net
    inference_with_pytorch(res2net_save_dir)

    #cnn:crnn:res2net = 1:1:1
    proba1 = fusion_model([cnn_save_dir, crnn_save_dir, res2net_save_dir], None)
    id1 = np.argmax(proba1, axis=1)
    sub.behavior_id = id1
    sub.to_csv('private_submit1.csv', index=False)

    #cnn:crnn:res2net = 0.38:0.34:0.28
    proba2 = fusion_model([cnn_save_dir, crnn_save_dir, res2net_save_dir], [0.38, 0.34, 0.28])
    id2 = np.argmax(proba2, axis=1)
    sub.behavior_id = id2
    sub.to_csv('private_submit2.csv', index=False)



if __name__ == '__main__':
    main()