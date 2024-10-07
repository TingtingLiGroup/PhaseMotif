import torch
from torch import nn
import numpy as np
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.conv_onehot_1 = nn.Conv2d(1, 64, (21, 2), 1, (0, 1))
        self.conv_alpha_1 = nn.Conv2d(1, 64, (9, 2), 1, (0, 1))
        self.conv_onehot_2 = nn.Conv2d(1, 64, (21, 3), 1, 0)
        self.conv_alpha_2 = nn.Conv2d(1, 64, (9, 3), 1, 0)
        self.conv_onehot_3 = nn.Conv2d(1, 64, (21, 5), 1, 0)
        self.conv_alpha_3 = nn.Conv2d(1, 64, (9, 5), 1, 0)
        self.conv_onehot_4 = nn.Conv2d(1, 64, (21, 15), 1, 0)
        self.conv_alpha_4 = nn.Conv2d(1, 64, (9, 15), 1, 0)

        self.fc_mu = nn.Sequential(
            nn.Linear(82, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 15),
        )
        self.fc_log_var = nn.Sequential(
            nn.Linear(82, 128),
            nn.ReLU(),
            nn.Linear(128, 32),
            nn.ReLU(),
            nn.Linear(32, 15),
        )

    def forward(self, one_hot):
        # print('one-hot',one_hot.shape)
        onehot2alpha_matrix = np.loadtxt('dicts/onehot2alphabet_matrix.txt', dtype=int)
        onehot2alpha_matrix = torch.from_numpy(onehot2alpha_matrix).float()
        onehot2alpha_matrix = onehot2alpha_matrix.cuda(0)

        # alpha数据
        # print(one_hot.device, onehot2alpha_matrix.device)
        alpha1 = torch.matmul(one_hot.float(), onehot2alpha_matrix)

        # 转置张量 加一个channel维度
        one_hot = one_hot.transpose(1, 2)
        one_hot = one_hot.unsqueeze(1)
        alpha = alpha1.transpose(1, 2)
        alpha = alpha.unsqueeze(1)

        # one_hot卷积
        one_hot_conv2 = self.conv_onehot_1(one_hot)
        one_hot_conv3 = self.conv_onehot_2(one_hot)
        one_hot_conv5 = self.conv_onehot_3(one_hot)
        one_hot_conv_all = self.conv_onehot_4(one_hot)

        # alpha卷积
        alpha_conv2 = self.conv_alpha_1(alpha)
        alpha_conv3 = self.conv_alpha_2(alpha)
        alpha_conv5 = self.conv_alpha_3(alpha)
        alpha_conv_all = self.conv_alpha_4(alpha)

        # 拼接
        all_conv = torch.cat([one_hot_conv2, one_hot_conv3, one_hot_conv5, one_hot_conv_all, alpha_conv2, alpha_conv3, alpha_conv5, alpha_conv_all], dim=-1)

        # 删掉1维度 + 全连接 (batch, 8, len)
        all_conv = all_conv.squeeze(2)
        # print('all_conv:', all_conv.shape)

        mu = self.fc_mu(all_conv)
        log_var = self.fc_log_var(all_conv)
        # print('mu:', mu.shape)

        return mu, log_var, alpha1


class Decoder(nn.Module):
    def __init__(self):
        super(Decoder, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, 8),
            nn.ReLU(),
            nn.Linear(8, 21),
        )

        self.conv = nn.Sequential(
            nn.Conv2d(64, 16, (1, 1), 1, (0, 0)),
            nn.ReLU(),
            nn.Conv2d(16, 4, (1, 1), 1, (0, 0)),
            nn.ReLU(),
            nn.Conv2d(4, 1, (1, 1), 1, (0, 0)),
        )

    def forward(self, z):
        # 加一个hang维度
        z = z.unsqueeze(-1)
        x_recon1 = self.fc(z)
        # print('x_recon1', x_recon1.shape)
        x_recon2 = self.conv(x_recon1)
        # print('x_recon1', x_recon1.shape)
        # print('x_recon2', x_recon2.shape)

        # 删除channel维度
        x_recon2 = x_recon2.squeeze(1)

        # sigmoid
        x_recon2 = torch.sigmoid(x_recon2)

        return x_recon2


class VAE(nn.Module):
    def __init__(self):
        super(VAE, self).__init__()
        self.encoder = Encoder()
        self.decoder = Decoder()

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x_list):
        # one_hot数据堆叠
        one_hot = torch.stack(x_list, dim=0)
        # print('one_hot', one_hot.shape)

        mu, log_var, alpha = self.encoder(one_hot)
        z = self.reparameterize(mu, log_var)
        # print('z', z.shape)
        x_recon = self.decoder(z)

        return one_hot, x_recon, mu, log_var, alpha