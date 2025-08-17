from typing import List
import numpy as np
import torch
import torch.nn as nn
import math
import torch.nn.functional as F


def preprocess(batched_inputs: List[torch.Tensor]):  # 对数据先填充
    """
        param:
            batched_inputs: 张量列表
        Return:
            padded_animos: 填充后的批量张量
            sizes_orig: 原始尺寸信息

    """
    # 保留原始尺寸
    sizes_orig = [[input.shape[-2], input.shape[-1]] for input in batched_inputs]
    # 找到最大尺寸
    max_size = max([max(size[0], size[1]) for size in sizes_orig])

    # 构造批量形状 (batch_size, channel, 固定, max_size)
    batch_shape = (len(batched_inputs), batched_inputs[0].shape[0], batched_inputs[0].shape[1], max_size)
    padded_animos = torch.Tensor().new_full(batch_shape, 0.0)
    for padded_animo, input in zip(padded_animos, batched_inputs):
        h, w = input.shape[1:]
        padded_animo[..., :h, :w].copy_(input)   # ... 来表示取剩余的所有维度

    return padded_animos, np.array(sizes_orig)


def postprocess(feature_map, size_orig, kernel, stride, padding):  # 卷积后再裁剪
    """
        param:
                feature_maps: 特征图张量
                size_orig: 原图尺寸
                kernel: 卷积核大小 tuple格式
                stride: 步长 int格式
                padding: 填充 tuple tuple格式
    """
    size_on_feature = [
        1,  # 确定是1维的东西
        int((size_orig[1] + 2 * padding[1] - 1 * (kernel - 1) - 1) / stride + 1)]
    crop = feature_map[:, :size_on_feature[0], :size_on_feature[1]]

    return crop


class FirstCNN(torch.nn.Module):  # 第一层CNN面临着不同的长度的序列 因此要对数据先填充 卷积后再裁剪获得有效范围 再送入SPPLayer
    def __init__(self, out_channel, kernel, stride, padding):  # Kernel只写列就好 Stride只写数就好
        super(FirstCNN, self).__init__()
        self.CNN_oneHot = nn.Conv2d(1, out_channel, kernel_size=(21, kernel), stride=(stride, stride), padding=padding)
        self.CNN_alphabet = nn.Conv2d(1, out_channel, kernel_size=(9, kernel), stride=(stride, stride), padding=padding)

    def forward(self, x):
        if x.shape[-2] == 21:
            feature_maps = self.CNN_oneHot(x)
        else:
            feature_maps = self.CNN_alphabet(x)
        return feature_maps


class AttentionLayer(torch.nn.Module):
    def __init__(self, num_head, head_size, value_size, input_size, kernel, stride, padding):
        """
        :param num_head: 几个头
        :param head_size: 单个K和Q的维度
        :param value_size: 单个V的维度
        :param input_size: 输入数据代表feature的维度
        :param kernel: 是上一层CNN的kernel
        :param stride: 是上一层CNN的stride
        :param padding: 是上一层CNN的padding
        """
        super(AttentionLayer, self).__init__()
        self.num_head = num_head
        self.head_size = head_size
        self.value_size = value_size

        self.multi_head_size = num_head * head_size  # m个头的先拼一起一起算 之后再拆开
        self.multi_value_size = num_head * value_size

        self.query = nn.Linear(input_size, self.multi_head_size)
        self.key = nn.Linear(input_size, self.multi_head_size)
        self.value = nn.Linear(input_size, self.multi_value_size)

        self.attn_dropout = nn.Dropout(0.5)  # 不一定用

        self.kernel = kernel
        self.stride = stride
        self.padding = padding

    def transpose_for_scores(self, x, isValue=False):  # 将multi拆开 value与k/q的维度可能不一样
        if isValue:     # 拆前（batch=1, n, multi_head_size） multi_head_size = num_head * head_size
            new_x_shape = x.size()[:-1] + (self.num_head, self.value_size)
        else:
            new_x_shape = x.size()[:-1] + (self.num_head, self.head_size)
        x = x.view(*new_x_shape)      # （batch=1, n, num_head, head_size）
        return x.permute(0, 2, 1, 3)  # （batch=1, num_head, n, head_size/value_size）方便后面 n, head_size相乘

    def attention_one_matrix(self, input_tensor):  # 期望的input_tensor为（batch=1, n, feature_size）
        # 但是实际的是 torch.Size([8, 1, 59]) 八个CNN核卷积结果的 1个特征的 59个位置  59为n 8为feature_size (1, 59, 8)
        input_tensor = input_tensor.permute(1, 2, 0)

        mixed_query_layer = self.query(input_tensor)  # m个头的先拼一起一起算KQV
        mixed_key_layer = self.key(input_tensor)
        mixed_value_layer = self.value(input_tensor)

        query_layer = self.transpose_for_scores(mixed_query_layer, isValue=False)  # 拆前（batch=1, n, multi_head_size）
        key_layer = self.transpose_for_scores(mixed_key_layer, isValue=False)  # 拆后（batch=1, num_head, n, head_size）
        value_layer = self.transpose_for_scores(mixed_value_layer, isValue=True)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.head_size)        # attention_score = Q*K'/sqrt(head_size)
        attention_probs = nn.Softmax(dim=-1)(attention_scores)  # 按行softmax标准化
        # attention_probs = self.attn_dropout(attention_probs)  # 不一定用

        context_layer = torch.matmul(attention_probs, value_layer)   # attention_score * V （batch=1, num_head, n,
        # value_size）
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()   # （batch=1, n, num_head, value_size）
        new_context_layer_shape = context_layer.size()[:-2] + (self.multi_value_size,)  # （batch=1, n, multi_value_size)
        context_layer = context_layer.view(*new_context_layer_shape)
        # 因为后面的SPP层又要 torch.Size([8, 1, 59])这样的 因此又得转换
        context_layer = context_layer.permute(2, 0, 1)

        return context_layer, attention_probs, value_layer

    def forward(self, input, sizes_orig):
        attention_result = []
        attention_probs_result = []
        attention_value = []
        for matrix, size in zip(input, sizes_orig):
            matrix = postprocess(matrix, size, self.kernel, self.stride, self.padding)
            temp, score, value = self.attention_one_matrix(matrix)
            attention_result.append(temp)
            attention_probs_result.append(score)
            attention_value.append(value)

        return attention_result, attention_probs_result, attention_value


class SPPLayer(nn.Module):
    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()
        self.num_levels = num_levels
        self.pool_type = pool_type
        self.relu = nn.ReLU()

    def spp_one_matrix(self, x):
        c, h, w = x.size()  # c:通道数 h:高 w:宽
        spp = None
        for i in range(self.num_levels):  # 一维的简化版情况，一般情况的金字塔代码在初稿CNN4IDR中
            level = i + 1
            kernel_size = math.ceil(w / level)
            stride = math.ceil(w / level)
            pooling = math.floor((kernel_size * level - w + 1) / 2)

            if pooling > 0.5 * kernel_size:  # 不被max_pool1d所允许，所以允许有重叠 -- 原创
                stride = kernel_size - 1
                kernel_size = w + stride - level * stride
                pooling = 0

            # 选择池化方式
            if self.pool_type == 'max_pool':
                tensor = F.max_pool1d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(h, -1)
            else:  # 否则就平均池化
                tensor = F.avg_pool1d(x, kernel_size=kernel_size, stride=stride, padding=pooling).view(h, -1)
            # 展开、拼接
            if i == 0:
                spp = tensor.view(h, c, -1)
            else:
                spp = torch.cat((spp, tensor.view(h, c, -1)), dim=2)
        return spp

    def forward(self, matrix_list):
        spp_result = None
        for matrix in matrix_list:
            if spp_result is None:
                spp_result = self.spp_one_matrix(matrix)
            else:
                spp_result = torch.cat((spp_result, self.spp_one_matrix(matrix)), dim=0)
        spp_result = self.relu(spp_result)
        return spp_result


class CNNFC(nn.Module):
    def __init__(self, cnn_in_channel):
        super(CNNFC, self).__init__()
        cnn_out_channel = 16
        kernel = 5
        self.cnn_pooling1 = nn.Sequential(
            nn.Conv1d(cnn_in_channel, cnn_out_channel, kernel, 1, 1),
            nn.MaxPool1d(2),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.cnn_pooling1(x)
        x = x.view(x.shape[0], -1)
        return x


class FC(nn.Module):
    def __init__(self, fc_in_channel):  # 等于拍平后的尺寸
        super(FC, self).__init__()
        self.fc1 = nn.Linear(fc_in_channel, 256, bias=True)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 1, bias=True)
        self.drop = nn.Dropout(0.5)
        self.batchNorm = nn.BatchNorm1d(256)

        nn.init.kaiming_uniform_(self.fc1.weight, mode='fan_in', nonlinearity='relu')
        nn.init.kaiming_uniform_(self.fc2.weight, mode='fan_in', nonlinearity='relu')

    def forward(self, x):
        x = self.drop(x)
        f1 = self.fc1(x)
        f1 = self.batchNorm(f1)
        f1 = self.relu(f1)
        f2 = self.fc2(f1)
        return f2


class PredictMain(nn.Module):
    def __init__(self, cnn1out_channel, cnn1kernel, cnn1stride, cnn1padding, num_head, head_size,
                 value_size, num_level):
        super(PredictMain, self).__init__()
        cnn1params = [cnn1kernel, cnn1stride, cnn1padding]
        self.FirstCNN = FirstCNN(cnn1out_channel, *cnn1params)
        self.Attention = AttentionLayer(num_head, head_size, value_size, cnn1out_channel, *cnn1params)
        self.SPPLayer = SPPLayer(num_levels=num_level, pool_type='max_pool')
        cnn2_in_channel = int(num_head * value_size)
        fc_in_channel = int(num_head * value_size * (num_level * (num_level + 1) / 2 - 2))
        self.CNNFC = CNNFC(cnn_in_channel=cnn2_in_channel)
        self.FC = FC(fc_in_channel=fc_in_channel)

    def forward(self, x1, x2, device):
        x1, sizes_orig = preprocess(x1)
        x2, _ = preprocess(x2)  # 宽一个是21一个是9 长度一样 保留长度就行
        x1 = x1.to(device)
        x2 = x2.to(device)

        combined = self.FirstCNN(x1) + self.FirstCNN(x2)
        attention, _, _ = self.Attention(combined, sizes_orig)
        spp_result = self.SPPLayer(attention)
        cnn_result = self.CNNFC(spp_result)
        fc_result = self.FC(cnn_result)
        return fc_result


class AnalyseMain(nn.Module):
    def __init__(self, cnn1out_channel, cnn1kernel, cnn1stride, cnn1padding, num_head, head_size,
                 value_size, num_level):
        super(AnalyseMain, self).__init__()
        cnn1params = [cnn1kernel, cnn1stride, cnn1padding]
        self.FirstCNN = FirstCNN(cnn1out_channel, *cnn1params)
        self.Attention = AttentionLayer(num_head, head_size, value_size, cnn1out_channel, *cnn1params)
        self.SPPLayer = SPPLayer(num_levels=num_level, pool_type='max_pool')
        cnn2_in_channel = int(num_head * value_size)
        fc_in_channel = int(num_head * value_size * (num_level * (num_level + 1) / 2 - 2))
        self.CNNFC = CNNFC(cnn_in_channel=cnn2_in_channel)
        self.FC = FC(fc_in_channel=fc_in_channel)

    def forward(self, x1, x2, device):
        x1, sizes_orig = preprocess(x1)
        x2, _ = preprocess(x2)  # 宽一个是21一个是9 长度一样 保留长度就行
        x1 = x1.to(device)
        x2 = x2.to(device)

        combined = self.FirstCNN(x1) + self.FirstCNN(x2)
        attention_result, attention_score, attention_value = self.Attention(combined, sizes_orig)
        spp_result = self.SPPLayer(attention_result)
        cnn_result = self.CNNFC(spp_result)
        fc_result = self.FC(cnn_result)
        return fc_result, attention_result, attention_score, attention_value


class Autoencoder(nn.Module):
    def __init__(self, input_dim):
        super(Autoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 20)
        )
        self.decoder = nn.Sequential(
            nn.Linear(20, 64),
            nn.ReLU(),
            nn.Linear(64, 128),
            nn.ReLU(),
            nn.Linear(128, input_dim),
            nn.Sigmoid()  # to ensure the output is between 0 and 1
        )

    def forward(self, x):
        encoded = self.encoder(x)
        x = self.decoder(encoded)
        return encoded, x






















































