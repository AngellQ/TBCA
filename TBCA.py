import torch.nn as nn
import torch
import math
import torch.nn.functional as F

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)

class ChannelGate(nn.Module):
    def __init__(self, gate_channel, reduction_ratio=16, num_layers=1):
        super(ChannelGate, self).__init__()
        #self.gate_activation = gate_activation
        self.gate_c = nn.Sequential()
        #add_module的功能为Module添加一个子module，对应名字为name。add_module(name, module)
        self.gate_c.add_module( 'flatten', Flatten())
        
        gate_channels = [gate_channel]
        #print('gate_channels0=',gate_channels)#[128]
        #" // " 表示整数除法，返回不大于结果的一个最大的整数。
        gate_channels += [gate_channel // reduction_ratio] * num_layers
        print('gate_channels1=',gate_channels)  #[128,8]      

        gate_channels += [gate_channel]
        #print('gate_channels2=',len(gate_channels))#3

        for i in range( len(gate_channels) - 2 ):
            self.gate_c.add_module( 'gate_c_fc_%d'%i, nn.Linear(gate_channels[i], gate_channels[i+1]) )
            print('gate_channels1Linear=',gate_channels)  #[128, 8, 128]


            self.gate_c.add_module( 'gate_c_bn_%d'%(i+1), nn.BatchNorm1d(gate_channels[i+1]) )
            print('gate_channels1BN=',gate_channels)  #[128, 8, 128]


            self.gate_c.add_module( 'gate_c_relu_%d'%(i+1), nn.ReLU() )
            print('gate_channels1RELU=',gate_channels)  #[128, 8, 128]

            self.gate_c.add_module( 'gate_c_fc_final', nn.Linear(gate_channels[-2], gate_channels[-1]) )
            print('gate_channels1Linear2=',gate_channels)  #[128, 8, 128]

            #self.sigmoid = nn.Sigmoid()

    def forward(self, in_tensor):
        # print(in_tensor.shape)#([64, 128, 1, 42])
        avg_pool = F.avg_pool2d( in_tensor, in_tensor.size(2), stride=in_tensor.size(2) )
        #print('avg_pool.shape',avg_pool.shape)#([64, 128, 1, 1])
        return self.gate_c( avg_pool ).unsqueeze(2).unsqueeze(3).expand_as(in_tensor)
        #print(in_tensor*M.shape)
        #print(in_tensor.shape,M.shape,(in_tensor*M).shape)
        #M = self.Sigmoid( in_tensor )
        #M = self.gate_c( in_tensor )
        #return in_tensor*M
       
class tbca(nn.Module):

    def __init__(self):
        super(tbca, self).__init__()
        #self.emb = nn.Embedding(5,emb_dim,padding_idx=0)
        self.convolution_seq_1 = nn.Sequential(
            # 普通卷积
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(4, 16), stride=(1, 2)),
            nn.ReLU(inplace=True),
            # BN标准化
            nn.BatchNorm2d(num_features=128)
        )
       
        self.convolution_shape_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=128, kernel_size=(5, 16), stride=(1, 2)),
            nn.ReLU(inplace=True),
            nn.BatchNorm2d(num_features=128)
        )
        
     
        self.ChannelGate_seq = ChannelGate(gate_channel=128, reduction_ratio=16, num_layers=1)
        self.ChannelGate_shape = ChannelGate(gate_channel=128, reduction_ratio=16, num_layers=1)
        
        self.max_pooling_1 = nn.Sequential(
            nn.MaxPool2d(kernel_size=(1, 2), stride=(1, 1)),
            nn.AdaptiveAvgPool2d(output_size=(1, 1)),
            )


        self.output = nn.Sequential(
            nn.AdaptiveMaxPool2d(output_size=(1, 1)),
            nn.Flatten(start_dim=1),
            nn.Linear(in_features=256, out_features=1),  # 64, 128, 256
            nn.Sigmoid()
        )
        

    # 前向传播
    def _forward_impl(self, seq, shape):

        seq = seq.float()
        #seq = self.emb(seq)#([64, 1, 16, 101])
        # print(seq.shape,shape.shape)
        #seq = seq.permute(0,1,3,2)
        shape = shape.float()
        # print(seq.shape,shape.shape)
        conv_seq_1 = self.convolution_seq_1(seq)
        #print('conv_seq_1.shape=',conv_seq_1.shape)#[64, 128, 1, 43]
        pool_seq_1 = self.max_pooling_1(conv_seq_1)
        #print('pool_seq_1.shape=',pool_seq_1.shape)#[64, 128, 1, 1]

        conv_shape_1 = self.convolution_shape_1(shape)
        #print('conv_shape_1.shape=',conv_shape_1.shape)#[64, 128, 1, 43]
        pool_shape_1 = self.max_pooling_1(conv_shape_1)
        #print('pool_shape_1.shape=',pool_shape_1.shape)#[64, 128, 1, 1]

        attention_seq_1 = self.ChannelGate_seq(pool_seq_1)
        #print('attention_seq_1.shape=',attention_seq_1.shape)#[64, 128, 64, 128])
        attention_shape_1 = self.ChannelGate_shape(pool_shape_1)
        #print('attention_shape_1.shape=',attention_shape_1.shape)#[64, 128, 64, 128])
        #o=torch.cat((attention_seq_1, attention_shape_1))
        #print('output',o.shape)
        return self.output(torch.cat((attention_seq_1, attention_shape_1), dim=1))


    def forward(self, seq, shape):
        return self._forward_impl(seq, shape)