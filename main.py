import torch
import math
import numpy as np
import torch.nn as nn
import random
import visdom
from model import MyModel
from Dataset import load_dataset

def initialize(layer):
    # Xavier_uniform will be applied to conv1d and dense layer
    if isinstance(layer,nn.Conv2d) or isinstance(layer, nn.Linear):
        torch.nn.init.xavier_uniform_(layer.weight.data)
        if layer.bias is not None:
            torch.nn.init.constant_(layer.bias.data, val = 0.0)


def ObtainRandom(length):
    list_info = []  # 定义一个空列表（用于接收产生的随机数）
    while True:
        info = random.randint(0,length-1) # 每次循环获取一次随机数
        if info not in list_info:  # 判断随机数是否在列表中
            list_info.append(info)  # 不在列表中 进行添加
        if len(list_info) ==length:  # 判断 长度是否达到100
            break  # 停止循环

    return list_info

def Catbatch(input_list, info_list, idx, batch_size, batch_num):


    if idx ==  (batch_num - 1):
        batch = info_list[idx * batch_size: ]
    else:
        batch = info_list[idx * batch_size: (idx + 1) * batch_size]


    for idx, x in enumerate(batch):

        if idx == 0:
            catbatch = torch.unsqueeze(input_list[x], dim=0)

        else:
            cattensor = torch.unsqueeze(input_list[x], dim=0)
            catbatch = torch.cat((catbatch, cattensor), dim = 0)


    return catbatch


viz = visdom.Visdom()
viz.line([0], [-1], win='train_loss', opts=dict(title='train_loss'))
viz.line([0], [-1], win='val_acc', opts=dict(title='val_acc'))

train_data_path = "1.tsv"
# train_data_path = ".\Data_DNA\\5hmC\\5hmC_H.sapiens\\train.tsv"
train_sequence_list, train_label_list, train_One_hot_matrix_input, train_NCP_matrix_input, train_DPCP_matrix_input, train_all_matrix_input = load_dataset(train_data_path)
train_list_info = ObtainRandom(len(train_sequence_list))

val_data_path = "1.tsv"
# val_data_path = ".\Data_DNA\\5hmC\\5hmC_H.sapiens\\test.tsv"
val_sequence_list, val_label_list,  val_One_hot_matrix_input, val_NCP_matrix_input, val_DPCP_matrix_input, val_all_matrix_input = load_dataset(val_data_path)
val_list_info = ObtainRandom(len(val_sequence_list))

device = torch.device('cuda')
torch.manual_seed(1234)
model = MyModel(13).to(device)
model.apply(initialize)
criteon = nn.CrossEntropyLoss().to(device)

batch_size = 16
patience, best_acc = 0, None
train_patience = 10
lr1 = 1e-2

for epoch in range(100):


    if (patience == train_patience):
        print("val_loss did not improve after {} Epochs, thus Earlystopping is calling".format(train_patience))
        break

    cnt, loss_sum = 0, 0
    val = epoch // 4
    lr = lr1 * pow(0.7, val)

    if lr < 1e-5:
        lr = 1e-5

    print(lr)

    batch_num = math.ceil(len(train_list_info) / batch_size)
    for idx in range(batch_num):


        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, dampening=0.5, weight_decay=0.01,nesterov=False)

        # x = Catbatch(train_One_hot_matrix_input, train_list_info, idx, batch_size, batch_num)
        # x = Catbatch(train_NCP_matrix_input, train_list_info, idx, batch_size, batch_num)
        # x = Catbatch(train_DPCP_matrix_input, train_list_info, idx, batch_size, batch_num)
        x = Catbatch(train_all_matrix_input, train_list_info, idx, batch_size, batch_num)
        label = Catbatch(train_label_list, train_list_info, idx, batch_size, batch_num)


        x, label = x.to(device), label.to(device)


        model.train()
        logits = model(x)
        loss = criteon(logits, label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_sum += loss
        cnt += 1

    final_loss = loss_sum / cnt

    viz.line([final_loss.item()], [epoch + 1], win='train_loss', update='append')
    print("Epoch: {}, Train_Loss: {}".format(epoch + 1, final_loss))

    if epoch % 1 == 0:

        model.eval()
        with torch.no_grad():
            cnt, total_correct = 0, 0
            batch_num = math.ceil(len(val_list_info) / batch_size)
            for idx in range(batch_num):


                # x = Catbatch(val_One_hot_matrix_input, val_list_info, idx, batch_size, batch_num)
                # x = Catbatch(val_NCP_matrix_input, val_list_info, idx, batch_size, batch_num)
                # x = Catbatch(val_DPCP_matrix_input, val_list_info, idx, batch_size, batch_num)
                x = Catbatch(val_all_matrix_input, val_list_info, idx, batch_size, batch_num)
                label = Catbatch(val_label_list, val_list_info, idx, batch_size, batch_num)
                x, label = x.to(device), label.to(device)

                logits = model(x)
                pred = logits.argmax(dim=1)

                correct = torch.eq(pred, label).int().sum().item()
                total_correct += correct
                cnt += x.size(0)

            acc = total_correct / cnt


            # Save best only
            if best_acc is None or acc > best_acc:
                best_acc, patience = acc, 0
                net_state_dict = model.state_dict()
                path_state_dict = "1.pt"
                torch.save(net_state_dict, path_state_dict)
            else:
                patience = patience + 1

            viz.line([acc], [epoch+1], win='val_acc', update='append')
            print("Epoch: {}, Valid_acc: {}".format(epoch + 1, acc))





