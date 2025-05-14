"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class CrossViewConLoss(nn.Module):

    def __init__(self, ):
        super(CrossViewConLoss, self).__init__()
        self.temperature = 1

    def forward(self, features):
        # feature的shape为B*3*d
        batch, view, dimension = features.size()
        # 1.首先对其进行转置和view操作 B*3*d-->3*B*d-->(3B)*d
        features = torch.permute(features, dims=(1, 0, 2)).contiguous().view(-1, dimension)
        # 2.计算余弦相似度矩阵
        x = features.unsqueeze(1)
        y = features.unsqueeze(0)
        cosine_similarities = torch.cosine_similarity(x, y, dim=-1)
        # 3.求绝对值
        abs_cosine = torch.abs(cosine_similarities)
        print('cosine_similarities', cosine_similarities)
        # 4.对每个位置求指数
        exp_cosine_similarities = torch.exp(cosine_similarities)
        exp_abs_cosine = torch.exp(abs_cosine)
        print('exp_cosine_similarities max', exp_cosine_similarities.max())
        print('exp_cosine_similarities min', exp_cosine_similarities.min())
        # 5.求分母(基于绝对值求分母)
        denominator = torch.sum(exp_abs_cosine, dim=-1)
        # 6.生成一个mask矩阵
        ones = torch.ones((batch, batch)).cuda()
        mask = torch.block_diag(ones, ones, ones)
        # 7.mask与指数后的相似度矩阵相乘，基于原始相似度求分子。
        masked_exp = mask * exp_cosine_similarities
        # 8.求分子
        numerator = torch.sum(masked_exp, dim=-1)
        # 9.分子除以分母然后取log，求和，除以batch，添加负号
        loss = -(torch.sum(torch.log(numerator / denominator)) / batch)

        return loss


def class_select(logits, target):
    # in numpy, this would be logits[:, target].
    # 450 2
    # 450表示patch数 2表示类别
    batch_size, num_classes = logits.size()
    if target.is_cuda:
        device = target.data.get_device()
        # torch.arange(0, 2).long().repeat(450, 1) 输出是450*2
        # target.data.repeat(num_classes, 1) 输出是2*450
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .cuda(device)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    else:
        one_hot_mask = torch.autograd.Variable(torch.arange(0, num_classes)
                                               .long()
                                               .repeat(batch_size, 1)
                                               .eq(target.data.repeat(num_classes, 1).t()))
    # 450*1
    # 根据one-hot将所属类别对应的logit提取出来.
    return logits.masked_select(one_hot_mask)


class LogitLoss(nn.Module):
    def __init__(self):
        super(LogitLoss, self).__init__()

    # alpha:3*150
    # logits:450*2
    # target:450
    # batch_size:3
    # ppt步骤3
    def forward(self, alpha, logits, target):
        # 模仿CrossEntropyLoss，先基于logit和target获取真实类别所对应的logit。
        # selected_logits:450*1->3*150
        # ppt步骤1 送入到class_select函数之前先进行softmax。
        alpha = alpha.view(alpha.size(0), alpha.size(1))
        print('alpha', alpha)
        softmax_logits = torch.softmax(logits, dim=1)
        print('softmax_logits', softmax_logits.shape)
        selected_logits = class_select(softmax_logits, target).view(alpha.shape[0], -1)
        print('selected_logits', selected_logits.shape)
        # 3*1
        # logits_sum = torch.sum(selected_logits, dim=1, keepdim=True)
        # print('logits_sum',logits_sum.shape)
        # 3*150
        # norm_logits = torch.div(selected_logits, logits_sum)
        # print('norm_logits',norm_logits.shape)
        # 3*150->3->1
        # logit_loss = torch.norm(input=alpha - norm_logits, p=2, dim=-1).sum()
        # logit_loss = torch.norm(input=torch.relu(torch.abs(alpha - selected_logits) - 0.2), p=2, dim=-1).sum()
        # print('logit_loss', logit_loss)

        logit_loss = torch.nn.functional.kl_div(torch.log_softmax(alpha, dim=-1),
                                                torch.softmax(selected_logits, dim=-1), reduction='batchmean')
        print('logit_loss', logit_loss)
        return logit_loss
