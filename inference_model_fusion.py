import numpy as np
import pandas as pd
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

def main():
    cnn_save_dir = "./save_dir/cnn"
    crnn_save_dir = "./save_dir/crnn"
    res2net_save_dir = "./save_dir/res2net"
    sub = pd.read_csv('./data/提交结果示例.csv')

    # cnn:crnn:res2net = 1:1:1
    proba1 = fusion_model([cnn_save_dir, crnn_save_dir, res2net_save_dir], None)
    id1 = np.argmax(proba1, axis=1)
    sub.behavior_id = id1
    sub.to_csv('inference_submit1.csv', index=False)

    # cnn:crnn:res2net = 0.38:0.34:0.28
    proba2 = fusion_model([cnn_save_dir, crnn_save_dir, res2net_save_dir], [0.38, 0.34, 0.28])
    id2 = np.argmax(proba2, axis=1)
    sub.behavior_id = id2
    sub.to_csv('inference_submit2.csv', index=False)

if __name__ == '__main__':
    main()