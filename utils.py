import numpy as np
import torch

mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
           4: 'D_4', 5: 'A_5', 6: 'B_1', 7: 'B_5',
           8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
           12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
           16: 'C_2', 17: 'C_5', 18: 'C_6'}

def to_cuda_if_available(*args):
    """ Transfer object (Module, Tensor) to GPU if GPU available
    Args:
        args: torch object to put on cuda if available (needs to have object.cuda() defined)

    Returns:
        Objects on GPU if GPUs available
    """
    res = list(args)
    if torch.cuda.is_available():
        for i, torch_obj in enumerate(args):
            res[i] = torch_obj.cuda()
    if len(res) == 1:
        return res[0]
    return res

def mixup_data(x, y, alpha=1.0, use_cuda=True):

    '''Compute the mixup data. Return mixed inputs, pairs of targets, and lambda'''
    if alpha > 0.:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1.
    batch_size = x.size()[0]
    if use_cuda:
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index,:]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def mixup_criterion(y_a, y_b, lam):
    return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def one_hot(labels, num_classes):
    labels = np.squeeze(labels)
    if labels.ndim==0:
        arr = np.zeros(num_classes)
        arr[labels]=1
        return arr
    batch_size = labels.shape[0]
    idxs = np.arange(0, batch_size, 1)
    arr = np.zeros([batch_size, num_classes])
    arr[idxs, labels] = 1
    return arr


def get_acc_combo():
    def combo(y, y_pred):
        # 数值ID与行为编码的对应关系
        mapping = {0: 'A_0', 1: 'A_1', 2: 'A_2', 3: 'A_3',
            4: 'D_4', 5: 'A_5', 6: 'B_1',7: 'B_5',
            8: 'B_2', 9: 'B_3', 10: 'B_0', 11: 'A_6',
            12: 'C_1', 13: 'C_3', 14: 'C_0', 15: 'B_6',
            16: 'C_2', 17: 'C_5', 18: 'C_6'}
        # 将行为ID转为编码

        code_y, code_y_pred = mapping[int(y)], mapping[int(y_pred)]
        if code_y == code_y_pred: #编码完全相同得分1.0
            return 1.0
        elif code_y.split("_")[0] == code_y_pred.split("_")[0]: #编码仅字母部分相同得分1.0/7
            return 1.0/7
        elif code_y.split("_")[1] == code_y_pred.split("_")[1]: #编码仅数字部分相同得分1.0/3
            return 1.0/3
        else:
            return 0.0
    confusionMatrix=np.zeros((19,19))
    for i in range(19):
        for j in range(19):
            confusionMatrix[i,j]=combo(i,j)
    def acc_combo(y, y_pred):
        y=np.argmax(y,axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        scores=confusionMatrix[y,y_pred]
        return np.mean(scores)
    return acc_combo

def get_acc_func():
    confusionMatrix=np.zeros((19,19))
    for i in range(19):
            confusionMatrix[i,i]=1
    def acc_func(y, y_pred):
        y=np.argmax(y,axis=1)
        y_pred = np.argmax(y_pred, axis=1)
        scores=confusionMatrix[y,y_pred]
        return np.mean(scores)
    return acc_func

