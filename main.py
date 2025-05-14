import os
import random
import time

import numpy as np

import argparse

import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold
from torch import nn
import os

from dataset import Traditional_Dataset
from losses import CrossViewConLoss, SimilarityLoss, LogitLoss

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

from torch.autograd import Variable
from torch.utils.data import DataLoader
from sklearn.metrics import confusion_matrix
from model import Baseline
from utils import cal_metrics, setup_seed

import setproctitle

setproctitle.setproctitle('22-2')


def train(epoch, optimizer, model):
    model.train()
    train_loss = 0
    total_correct = 0
    total = 0
    # 开始时间
    start = time.perf_counter()
    for batch_idx, (data, bag_label) in enumerate(train_loader):
        # print('data',data.shape)
        # print('bag_label',bag_label)
        if not args.no_cuda:
            data, bag_label = data.to(device), bag_label.to(device)
        data, bag_label = Variable(data), Variable(bag_label)
        # reset gradients
        optimizer.zero_grad()
        out1, attention_related, multiview_related = model(data)

        attention1, attention2, attention3, slice_logits1, slice_logits2, slice_logits3 = attention_related
        private_fusion, view1, view2, view3, diff1, diff2, diff3 = multiview_related

        loss1 = criterion(out1, bag_label)
        view1_loss = criterion(view1, bag_label)
        view2_loss = criterion(view2, bag_label)
        view3_loss = criterion(view3, bag_label)
        # private_fusion:B*3*d
        loss2 = conloss(features=private_fusion)
        # common_loss = simloss(x=common_fusion)

        common1_loss = criterion(diff1, bag_label)
        common2_loss = criterion(diff2, bag_label)
        common3_loss = criterion(diff3, bag_label)
        common_loss = common1_loss + common2_loss + common3_loss

        # 这里的2三分类和四分类要记得修改😍
        attention_loss1 = logit_loss(alpha=attention1, logits=slice_logits1.view(-1, 2),
                                     target=bag_label.repeat(slice_logits1.size(1), 1).permute(1, 0).contiguous().view(
                                         -1))
        attention_loss2 = logit_loss(alpha=attention2, logits=slice_logits2.view(-1, 2),
                                     target=bag_label.repeat(slice_logits2.size(1), 1).permute(1, 0).contiguous().view(
                                         -1))
        attention_loss3 = logit_loss(alpha=attention3, logits=slice_logits3.view(-1, 2),
                                     target=bag_label.repeat(slice_logits3.size(1), 1).permute(1, 0).contiguous().view(
                                         -1))

        attention_loss = attention_loss1 + attention_loss2 + attention_loss3
        print('loss1', loss1)
        print('loss2', loss2)
        print('view1_loss', view1_loss)
        print('view2_loss', view2_loss)
        print('view3_loss', view3_loss)
        print('common_loss', common_loss)
        # print('commonxx_loss', commonxx_loss)
        # loss = loss1 + loss2 + loss3
        loss = loss1 + 0.4*loss2 + (view1_loss + view2_loss + view3_loss) + 0.4*common_loss + attention_loss
        print('out1' + str(out1) + 'bag_label' + str(bag_label))
        # 损失累加变相增加batchsize，新增
        # 反向传播，计算梯度
        loss.backward()
        # 更新参数
        optimizer.step()
        train_loss += loss.item()
        _, predicted = out1.max(1)
        total_correct += predicted.eq(bag_label).sum().item()
        total += bag_label.size(0)
    # 结束时间
    end = time.perf_counter()
    # calculate loss and error for epoch
    train_loss /= len(train_loader)
    total_correct /= total
    # 指标记录
    content.append(str(end - start))
    content.append(str(train_loss))
    content.append(str(total_correct))
    print('Epoch: {}, Loss: {:.4f}, Total correct: {:.4f}'.format(epoch, train_loss, total_correct))


def test(model):
    model.eval()
    global best_acc
    test_loss = 0
    correct = 0
    total = 0
    target_y = np.array([])
    pred_y = np.array([])
    start = time.perf_counter()
    with torch.no_grad():
        for batch_idx, (data, bag_label) in enumerate(test_loader):
            if not args.no_cuda:
                data, bag_label = data.to(device), bag_label.to(device)
            data, bag_label = Variable(data), Variable(bag_label)
            out1, _, _ = model(data)
            loss1 = criterion(out1, bag_label)
            print('loss1', loss1)
            # print('loss3', loss3)
            # loss = loss1 + loss2 + loss3
            loss = loss1
            print('out1' + str(out1) + 'bag_label' + str(bag_label))
            test_loss += loss.item()
            _, predicted = out1.max(1)
            target_y = np.append(target_y, bag_label.tolist())
            pred_y = np.append(pred_y, predicted.tolist())
            correct += predicted.eq(bag_label).sum().item()
            total += bag_label.size(0)
    test_accuracy = correct / total
    test_loss /= len(test_loader)
    # 结束时间
    end = time.perf_counter()
    content.append(str(end - start))
    content.append(str(test_loss))
    content.append(str(test_accuracy))
    cf_matrix = confusion_matrix(target_y, pred_y)
    metrics_result, count_record = cal_metrics(np.array(cf_matrix))
    # 两个类别各自的precision
    content.append(str(metrics_result[0][0]))
    content.append(str(metrics_result[1][0]))
    # 两个类别各自的sensitivity
    content.append(str(metrics_result[0][1]))
    content.append(str(metrics_result[1][1]))
    # 两个类别各自的specificity
    content.append(str(metrics_result[0][2]))
    content.append(str(metrics_result[1][2]))
    # 两个类别各自的accuracy
    content.append(str(metrics_result[0][3]))
    content.append(str(metrics_result[1][3]))
    # 两个类别各自的f1_score
    content.append(str(metrics_result[0][4]))
    content.append(str(metrics_result[1][4]))

    res = pd.DataFrame(columns=column)
    res.loc[0] = content
    # print('content', content)
    cur_dir = os.path.dirname(__file__)
    if os.path.exists(os.path.join(cur_dir, "cls_metrics.xlsx")):
        df = pd.read_excel(os.path.join(cur_dir, "cls_metrics.xlsx"), index_col=0)
        res = pd.concat([df, res], ignore_index=True)
        res.to_excel(os.path.join(cur_dir, "cls_metrics.xlsx"))
    else:
        res.to_excel(os.path.join(cur_dir, "cls_metrics.xlsx"))
    content.clear()

    # Save checkpoint.
    if test_accuracy > best_acc:
        print('Saving..')
        state = {
            'net': model.state_dict(),
            'acc': test_accuracy,
            'epoch': epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state,
                   './checkpoint/' + 'fold' + str(fold) + '_epoch' +
                   str(epoch) + '_accuracy' + str(test_accuracy) + '.pth')
        best_acc = test_accuracy
    print('\nTest Set, Loss: {:.4f}, Test accuracy: {:.4f}'.format(test_loss, test_accuracy))


if __name__ == "__main__":
    print('Start Training')
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST bags Example')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 20)')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--reg', type=float, default=10e-4, metavar='R',
                        help='weight decay')
    parser.add_argument('--seed', type=int, default=3, metavar='S',
                        help='random seed (default: 3)')
    parser.add_argument('--device', type=int, default=0, metavar='D',
                        help='gpu (default: 0)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--model', type=str, default='attention', help='Choose b/w attention and gated_attention')

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.cuda.set_device(args.device)
    print('当前使用的显卡', torch.cuda.current_device())
    # 设置随机数种子
    # torch.use_deterministic_algorithms(True)
    setup_seed(args.seed)
    print('Load Train and Test Set')

    data_list = []
    labels_list = []
    with open(r'AD_NC_index.txt', encoding="utf-8") as file:
        content = file.readlines()
        # 逐行读取数据
        for line in content:
            data_list.append(line.split('   ')[0])
            labels_list.append(line.split('   ')[1].replace('\n', ''))
    data_list = np.array(data_list)
    labels_list = np.array(labels_list)
    print('数组集总长度', len(data_list))
    # , random_state=1, shuffle=True
    skf = StratifiedKFold(n_splits=5, random_state=3, shuffle=True)
    # 5折交叉验证
    fold = 0
    for train_index, test_index in skf.split(data_list, labels_list):
        # if fold < 1:
        #   fold = fold + 1
        #    continue
        trainset = Traditional_Dataset(data_list=data_list[train_index], label_list=labels_list[train_index],
                                       is_training=True)
        testset = Traditional_Dataset(data_list=data_list[test_index], label_list=labels_list[test_index],
                                      is_training=False)
        print('交叉验证' + str(fold) + '训练集长度' + str(len(trainset)))
        print('交叉验证' + str(fold) + '测试集长度' + str(len(testset)))
        # 当使用dataset_patch时batch_size要设置为1
        train_loader = DataLoader(trainset, batch_size=4, shuffle=True)
        test_loader = DataLoader(testset, batch_size=1, shuffle=False)
        best_acc = 0
        print('Init Model')
        model = Baseline()
        if not args.no_cuda:
            model.to(device)
        criterion = nn.CrossEntropyLoss()
        conloss = CrossViewConLoss()
        simloss = SimilarityLoss()
        logit_loss = LogitLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
        # optimizer = torch.optim.SGD(model.parameters(), momentum=0.9, lr=args.lr, weight_decay=args.reg)
        # optimizer = Adadelta(model.parameters(), lr=args.lr)
        # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 40, 60, 80], gamma=0.5,
        #                                                  verbose=True)
        column = (
            'train_epoch_time', 'train_loss', 'train_accuracy',
            'test_epoch_time', 'test_loss', 'test_accuracy',
            'precision_AD', 'precision_NC',
            'sensitivity_AD', 'sensitivity_NC',
            'specificity_AD', 'specificity_NC',
            'accuracy_AD', 'accuracy_NC',
            'F1_AD', 'F1_NC')

        content = []

        for epoch in range(1, args.epochs + 1):
            train(epoch, optimizer, model=model)
            print('Start Testing')
            test(model)
            # scheduler.step()

        fold = fold + 1
