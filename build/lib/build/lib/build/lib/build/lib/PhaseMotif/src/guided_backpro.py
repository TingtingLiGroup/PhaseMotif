import torch
import torch.nn as nn
import pandas as pd

from ..utils.analyse import explain_afterbgp, explain_Spp, explain_Attention, explain_firstCNN, important_position2animo, \
    explain_firstCNN_feature


class GuidedBackprop:
    def __init__(self, model):
        self.model = model
        self.model.eval()
        self.info = None
        self.activation_maps = []

        self.hooks = []
        self.register_hooks()

    def remove_hooks(self):
        if self.hooks == []:
            return
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def register_hooks(self):

        def first_layer_hook_fn(module, grad_in, grad_out):
            # 在全局变量中保存输入图片的梯度，该梯度由第一层卷积层
            # 反向传播得到，因此该函数需绑定池化relu层
            # print('first_layer', grad_in[0])
            self.info = grad_in[0]

        def forward_hook_fn(module, input, output):
            # 在全局变量中保存 ReLU 层的前向传播输出
            # 用于将来做 guided backpropagation
            self.activation_maps.append(output)

        def backward_hook_fn(module, grad_in, grad_out):
            # ReLU 层反向传播时，用其正向传播的输出作为 guide
            # 反向传播和正向传播相反，先从后面传起
            grad = self.activation_maps.pop()
            # ReLU 正向传播的输出要么大于0，要么等于0.大于 0 的部分，梯度为1; 等于0的部分，梯度还是 0
            grad[grad > 0] = 1

            # grad_out[0] 表示 feature 的梯度，只保留大于 0 的部分
            positive_grad_out = torch.clamp(grad_out[0], min=0.0)
            # 创建新的输入端梯度
            new_grad_in = positive_grad_out * grad

            # ReLU 不含 parameter，输入端梯度是一个只有一个元素的 tuple
            # print(new_grad_in)
            return (new_grad_in,)

        # 获取 module
        modules = list(self.model.named_modules())

        # 遍历所有 module，对 ReLU 注册 forward hook 和 backward hook
        for name, module in modules:
            if isinstance(module, nn.ReLU):
                handle1 = module.register_forward_hook(forward_hook_fn)
                handle2 = module.register_full_backward_hook(backward_hook_fn)
                self.hooks.append(handle1)
                self.hooks.append(handle2)

        # 对卷积层前的金字塔池化完注册 hook
        first_layer = modules[10][1]
        # print('first_layer', modules[10])
        handle3 = first_layer.register_full_backward_hook(first_layer_hook_fn)
        self.hooks.append(handle3)

    def visualize(self, loader, device, divide=-1, feature=False):  # 传入的dataloader batch一定是1
        position_list = []  # 存放重要特征位点的位置
        feauture_list = []  # 存放重要特征的序列

        for i, (data_oneHot, data_alphabet, label, protein) in enumerate(loader):
            if divide != -1:
                if divide * 2000 <= i < (divide + 1) * 2000:
                    # print('i', i)
                    pass
                else:
                    continue
            else:
                if protein[0] not in [10514, 2497, 3406, 4994, 3964, 11357, 8156, 11465, 3908, 5206, 3399, 13044, 3516,
                                      4000, 1623, 5562, 4612]:
                    continue

            fc_result, attention_result, attention_score, attention_value = self.model(data_oneHot, data_alphabet,
                                                                                       device)
            if fc_result[0].item() >= 0 and label[0] == 1:  # 满足条件再继续找关键序列
                # 获取输出，之前注册的 forward hook 开始起作用
                # 反向传播，之前注册的 backward hook 开始起作用
                fc_result.backward()
                # 得到反向梯度值
                dgp = self.info.data[0]
                dgp1 = dgp.detach().cpu().numpy()
                # 数据从tensor转换出来
                attention_result1 = attention_result[0].detach().cpu().numpy().reshape(*attention_result[0].shape[-3:])
                attention_score1 = attention_score[0].detach().cpu().numpy().reshape(*attention_score[0].shape[-3:])
                attention_value1 = attention_value[0].detach().cpu().numpy().reshape(*attention_value[0].shape[-3:])
                # 重要位点
                imp_pos1, imp_value_dic = explain_afterbgp(dgp1)
                imp_pos2 = explain_Spp(before_SPPData=attention_result1,
                                       need_index=imp_pos1, gradient_dic=imp_value_dic, n_large=-1)

                if feature is True:  # if you want a density TU
                    all_position, all_times = explain_Attention(attention_score=attention_score1,
                                                                attention_value=attention_value1,
                                                                need_index=imp_pos2, n_large1=20,
                                                                n_large2=-1)
                    feature_result, start_position = explain_firstCNN_feature(CNN_padding=1, CNN_Stride=1, CNN_kernel=15,
                                                              need_index=all_position,
                                                              index_times=all_times,
                                                              dataRealLen=data_oneHot[0].shape[-1])

                    feauture_list.append([protein[0], feature_result])
                    position_list.append([protein[0], start_position])
                    continue

                imp_pos3 = explain_Attention(attention_score=attention_score1, attention_value=attention_value1,
                                             need_index=imp_pos2, n_large1=20,
                                             n_large2= int(data_oneHot[0].shape[-1] / 30) + 1)  # -1 means all  now use
                imp_pos4 = explain_firstCNN(CNN_padding=1, CNN_Stride=1, CNN_kernel=15, need_index=list(imp_pos3),
                                            dataRealLen=data_oneHot[0].shape[-1])

                seq = important_position2animo(imp_pos4, data_oneHot[0][0].detach().cpu().numpy())

                temp_feature = [protein[0], seq]
                temp_position = [protein[0], sorted(imp_pos4)]
                position_list.append(temp_position)
                feauture_list.append(temp_feature)

        feauture_list = pd.DataFrame(feauture_list, columns=['Gene', 'Feature'])
        position_list = pd.DataFrame(position_list, columns=['Gene', 'Position'])
        return feauture_list, position_list
