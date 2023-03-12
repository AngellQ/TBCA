 # -*- coding: utf-8 -*
import os

os.environ["CUDA_VISIBLE_DEVICES"] = '0'

import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as loader
import math
import numpy as np

from tqdm import tqdm
from sklearn.metrics import accuracy_score, roc_auc_score, precision_recall_curve, auc
from torch.utils.data import random_split
from Dataset.scripts import SSDataset_690


class Constructor:
   

    def __init__(self, model, model_name='tbca'):

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(device=self.device)
        self.model_name = model_name
        #Adam()算法通过改善训练方式，来最小化(或最大化)损失函数，从而调整模型更新权重和偏差参数
        self.optimizer = optim.Adam(self.model.parameters())
        #学习率调整
        self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer=self.optimizer, patience=5, verbose=1)
        #nn.BCELoss 用于计算预测值和真实值之间的二元交叉熵损失，默认对一个batch里面的数据做二元交叉熵并且求平均。
        self.loss_function = nn.BCELoss()

        self.batch_size = 64
        self.epochs = 30
        
    def learn(self, TrainLoader, ValidateLoader):

        path = os.path.abspath(os.curdir)

        for epoch in range(self.epochs):
            self.model.train()
            #tqdm是一个显示进度条的python工具包
            ProgressBar = tqdm(TrainLoader)
            for data in ProgressBar:
                self.optimizer.zero_grad()
                #zero_grad() 将梯度清零
                
                ProgressBar.set_description("Epoch %d" % epoch)
                seq, shape, label = data
                # print(seq.shape, shape.shape, label[0])
                #.to(device) 可以指定CPU 或者GPU
                output = self.model(seq.unsqueeze(1).to(self.device), shape.unsqueeze(1).to(self.device))
                #output = self.model(seq.unsqueeze(1), shape.unsqueeze(1))

                loss = self.loss_function(output, label.float().to(self.device))
                #显示进度情况
                ProgressBar.set_postfix(loss=loss.item())
                #.backward()自动计算所有的梯度，实现反向传播
                loss.backward()
                self.optimizer.step()

            valid_loss = []
            #评估模式
            self.model.eval()
            #torch.no_grad()是一个上下文管理器,被该语句 wrap 起来的部分将不会track 梯度
            #训练模型
            with torch.no_grad():
                for valid_seq, valid_shape, valid_labels in ValidateLoader:
                    valid_output = self.model(valid_seq.unsqueeze(1).to(self.device), valid_shape.unsqueeze(1).to(self.device))
                    valid_labels = valid_labels.float().to(self.device)

                    valid_loss.append(self.loss_function(valid_output, valid_labels).item())

                valid_loss_avg = torch.mean(torch.Tensor(valid_loss))
                self.scheduler.step(valid_loss_avg)

        torch.save(self.model.state_dict(), path + '/' + self.model_name + '.pth')

        print('\n---Finish Learn---\n')

    def inference(self, TestLoader):

        path = os.path.abspath(os.curdir)
        self.model.load_state_dict(torch.load(path + '/' + self.model_name + '.pth', map_location='cpu'))

        predicted_value = []
        ground_label = []
        self.model.eval()

        for seq, shape, label in TestLoader:
            output = self.model(seq.unsqueeze(1).to(self.device), shape.unsqueeze(1).to(self.device))
            """ To scalar"""
            predicted_value.append(output.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy())
            ground_label.append(label.squeeze(dim=0).squeeze(dim=0).detach().cpu().numpy())

        print('\n---Finish Inference---\n')

        return predicted_value, ground_label

    def measure(self, predicted_value, ground_label):
        accuracy = accuracy_score(y_pred=np.array(predicted_value).round(), y_true=ground_label)
        roc_auc = roc_auc_score(y_score=predicted_value, y_true=ground_label)

        precision, recall, _ = precision_recall_curve(probas_pred=predicted_value, y_true=ground_label)
        #print('precision=',precision, 'recall=',recall)
        pr_auc = auc(recall, precision)

        print('\n---Finish Measure---\n')
        print('acc=',accuracy, 'roc_auc=',roc_auc,  'pr_auc=',pr_auc)

        return accuracy, roc_auc, pr_auc

    def run(self, samples_file_name, ratio=0.8):

        Train_Validate_Set = SSDataset_690(samples_file_name, False)

        """divide Train samples and Validate samples"""
        #随机划分后对划分后数据处理
        Train_Set, Validate_Set = random_split(dataset=Train_Validate_Set,
                                               lengths=[math.ceil(len(Train_Validate_Set) * ratio),
                                                        len(Train_Validate_Set) -
                                                        math.ceil(len(Train_Validate_Set) * ratio)],
                                               generator=torch.Generator().manual_seed(0))

        TrainLoader = loader.DataLoader(dataset=Train_Set, drop_last=True,
                                        batch_size=self.batch_size, shuffle=True, num_workers=0)
        ValidateLoader = loader.DataLoader(dataset=Validate_Set, drop_last=True,
                                           batch_size=self.batch_size, shuffle=False, num_workers=0)

        TestLoader = loader.DataLoader(dataset=SSDataset_690(samples_file_name, True),
                                       batch_size=1, shuffle=False, num_workers=0)

        self.learn(TrainLoader, ValidateLoader)
        predicted_value, ground_label = self.inference(TestLoader)

        accuracy, roc_auc, pr_auc = self.measure(predicted_value, ground_label)

        print('\n---Finish Run---\n')
        print(accuracy,roc_auc,pr_auc)

        return accuracy, roc_auc, pr_auc
     


from TBCA import tbca

Train = Constructor(model=tbca())

Train.run(samples_file_name='wgEncodeAwgTfbsBroadDnd41CtcfUniPk')
