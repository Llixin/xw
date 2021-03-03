from dataset import  XWDataset
from metrics import  XWMetrics
from torch.utils.data import DataLoader
from utils import  one_hot
from torch_func import  Agent

from torch.optim import lr_scheduler
from torch.optim import Adam as torch_adam
import torch
import os
import numpy as np
import pandas as pd
from Model.Model_Res2Net import Res2Net, CELoss
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


def inference_with_pytorch(save_dir):
    """
    Train Res2Net with pytorch
    Using Normalization
    Data augmentation: jitter, permutation
    Loss: categorical_crossentropy
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    sub = pd.read_csv('./data/提交结果示例.csv')
    NUM_CLASSES = 19
    METRICS=XWMetrics()

    test_data=XWDataset("./data/sensor_test.csv",with_label=False)
    proba_t = np.zeros((len(test_data), NUM_CLASSES))
    folds=10
    seeds = [1, 2, 3]
    valid_acc_list = []
    count = 0
    # EPOCH = 100
    BATCH_SIZE = 256
    for seed in seeds:
        train_data = XWDataset("./data/sensor_train.csv", with_label=True)
        save_model_dir=os.path.join(save_dir, 'seed'+str(seed))
        if not os.path.exists(save_model_dir):
            os.makedirs(save_model_dir)
        train_data.stratifiedKFold(seed, folds)
        for fold in range(folds):
            model=Res2Net(num_classes=NUM_CLASSES)
            save_model_fold_dir=os.path.join(save_model_dir,"fold_{}".format(fold))
            agent=Agent(model=model,device_info=DEVICE_INFO, save_dir=save_model_fold_dir)
            earlyStopping = None

            LOSS={ "celoss":(CELoss(), 1.0)}
            #LOSS_WEIGHT = {"celoss": 1.0, "centerloss": 0.005}
            OPTIM=torch_adam(model.parameters(), lr=3e-4)
            reduceLR = lr_scheduler.ReduceLROnPlateau(OPTIM, mode="max", factor=0.5, patience=8, verbose=True, min_lr=1e-5)
            agent.compile(loss_dict=LOSS,optimizer=OPTIM, metrics=METRICS)
            agent.summary()
            valid_X,valid_Y=train_data.get_valid_data(fold)
            valid_Y=one_hot(valid_Y,NUM_CLASSES)
            valid_data = [(valid_X[i],valid_Y[i]) for i in range(valid_X.shape[0])]

            train_generator=DataLoader(train_data,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)
            # agent.fit_generator(train_generator, epochs=EPOCH,
            #                               validation_data=valid_data,
            #                         reduceLR=reduceLR,
            #                         earlyStopping=earlyStopping)

            agent.load_best_model()
            valid_pred = agent.predict(valid_X, batch_size=1024,phase="valid")
            print(valid_pred.shape)
            acc = cul_acc_combo(np.argmax(valid_Y, axis=1), np.argmax(valid_pred, axis=1))
            print(acc)
            valid_acc_list.append(acc)

            test_X=[test_data.data[i] for i in range(test_data.data.shape[0])]
            scores_test= agent.predict(test_X,batch_size=1024,phase="test")
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

def inference_res2net():
    res2net_save_dir = "./save_dir/res2net"

    # inference res2net
    inference_with_pytorch(res2net_save_dir)

if __name__ == '__main__':
    inference_res2net()