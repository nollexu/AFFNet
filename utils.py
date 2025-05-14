# patch_size=pad*2+1
import random

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F



def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

def cal_metrics(cf_matrix):
    n_classes = cf_matrix.shape[0]
    metrics_result = []
    count_record = []
    for i in range(n_classes):
        # 逐步获取 真阳，假阳，真阴，假阴四个指标，并计算三个参数
        ALL = np.sum(cf_matrix)
        # 对角线上是正确预测的
        TP = cf_matrix[i, i]
        # 列加和减去正确预测是该类的假阳
        FP = np.sum(cf_matrix[:, i]) - TP
        # 行加和减去正确预测是该类的假阴
        FN = np.sum(cf_matrix[i, :]) - TP
        # 全部减去前面三个就是真阴
        TN = ALL - TP - FP - FN
        # 0号位置是Precision✔
        # 1号位置是Sensitivity=recall✔
        # 2号位置是Specificity✔
        # 3号位置是accuracy✔
        # 4号位置为f1_score
        precision = (TP / (TP + FP))
        sensitivity = (TP / (TP + FN))
        specificity = (TN / (TN + FP))
        accuracy = (TN + TP) / ALL
        if TP + FP == 0:
            precision = 0
        if TP + FN == 0:
            sensitivity = 0
        if TN + FP == 0:
            specificity = 0
        if ALL == 0:
            accuracy = 0
        if precision == 0 or sensitivity == 0:
            f1_score = 0
        else:
            f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
        metrics_result.append([precision, sensitivity, specificity, accuracy, f1_score])
        count_record.append([TP, FP, FN, TN])
    return metrics_result, count_record