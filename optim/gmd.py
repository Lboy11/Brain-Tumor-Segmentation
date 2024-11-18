import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import pdb
import numpy as np
import copy
import random


class GMD:
    def __init__(self, optimizer, reduction='mean', writer=None):
        self._optim, self._reduction = optimizer, reduction
        self.iter = 0
        self.writer = writer

    @property
    def optimizer(self):
        return self._optim

    def zero_grad(self):
        '''
        clear the gradient of the parameters
        '''

        return self._optim.zero_grad(set_to_none=True)

    def step(self):
        '''
        update the parameters with the gradient
        '''

        return self._optim.step()

    def pc_backward(self, objectives, ddp_model=None):
        '''
        计算参数的梯度
        输入：
        - objectives: 目标函数列表
        '''

        # 将梯度打包成列表
        grads, shapes, has_grads = self._pack_grad(objectives, ddp_model)
        # print("grads", len(grads))
        # print("grads==", grads[0].shape)
        # print("has_grads", len(has_grads))
        # print("has_grads==", has_grads[0].shape)
        # 处理梯度冲突，调整梯度方向
        pc_grad = self._project_conflicting(grads, has_grads)
        # 将处理后的梯度恢复成原始形状
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        # 将处理后的梯度设置到模型参数中
        self._set_grad(pc_grad)
        return

    def pc_backward1(self, objectives, ddp_model=None):
        '''
        calculate the gradient of the parameters
        input:
        - objectives: a list of objectives
        '''

        grads, shapes, has_grads = self._pack_grad(objectives, ddp_model)
        pc_grad = self._project_conflicting(grads, has_grads)
        pc_grad = self._unflatten_grad(pc_grad, shapes[0])
        self._set_grad(pc_grad)
        return


    def _project_conflicting(self, grads, has_grads, shapes=None):
        # 创建共享参数的掩码，表示是否所有任务都具有梯度
        shared = torch.stack(has_grads).prod(0).bool()
        # 深拷贝梯度列表，并记录梯度的数量
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        # 初始化每个任务的系数
        coefs = torch.ones(num_task, dtype=torch.float32, device=grads[0].device)
        # 对每对梯度进行处理，解决梯度冲突
        for g_i in pc_grad:
            # 随机打乱梯度顺序
            indices = list(range(num_task))
            random.shuffle(list(range(num_task)))
            random.shuffle(grads)
            for index in indices:
                g_j = grads[index]
                # 计算两个梯度之间的点积
                g_i_g_j = torch.dot(g_i, g_j)
                # 如果两个梯度之间的点积小于0，则表示存在冲突
                if g_i_g_j < 0:
                    # 计算修正系数
                    coef = g_i_g_j / (g_j.norm() ** 2)
                    # 更新当前梯度，并更新系数
                    g_i -= coef * g_j
                    coefs[index] -= coef
        # 初始化合并后的梯度
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)
        # 记录迭代次数
        self.iter += 1
        # 将每个任务的系数写入 TensorBoard 中
        for ii, coef in enumerate(coefs):
            self.writer.add_scalar(f'coef/pc_grad_coef_{ii}', coef.item(), self.iter)
        # 根据指定的减少方法合并梯度
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared] for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')
        # 对非共享参数合并梯度
        merged_grad[~shared] = torch.stack([g[~shared] for g in pc_grad]).sum(dim=0)
        # 返回合并后的梯度
        return merged_grad


    def _project_conflicting1(self, grads, has_grads, shapes=None):
        shared = torch.stack(has_grads).prod(0).bool()
        pc_grad, num_task = copy.deepcopy(grads), len(grads)
        coefs = torch.ones(num_task, dtype=torch.float32, device=grads[0].device)
        for g_i in pc_grad:
            indices = list(range(num_task))
            random.shuffle(list(range(num_task)))
            random.shuffle(grads)
            for index in indices:
                g_j = grads[index]
                g_i_g_j = torch.dot(g_i, g_j)
                if g_i_g_j < 0:
                    coef = g_i_g_j / (g_j.norm() ** 2)

                    g_i -= coef * g_j
                    coefs[index] -= coef
        merged_grad = torch.zeros_like(grads[0]).to(grads[0].device)

        self.iter += 1
        for ii, coef in enumerate(coefs):
            self.writer.add_scalar(f'coef/pc_grad_coef_{ii}', coef.item(), self.iter)
        if self._reduction:
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).mean(dim=0)
        elif self._reduction == 'sum':
            merged_grad[shared] = torch.stack([g[shared]
                                               for g in pc_grad]).sum(dim=0)
        else:
            exit('invalid reduction method')

        merged_grad[~shared] = torch.stack([g[~shared]
                                            for g in pc_grad]).sum(dim=0)
        return merged_grad

    def _set_grad(self, grads):
        '''
        将修改后的梯度设置回网络参数中
        '''
        idx = 0
        # 遍历优化器的参数组
        for group in self._optim.param_groups:
            # 遍历参数组中的参数
            for p in group['params']:
                # 将对应位置的梯度设置为修改后的梯度
                p.grad = grads[idx]
                # 更新索引，指向下一个参数的梯度
                idx += 1
        return


    def _set_grad1(self, grads):
        '''
        set the modified gradients to the network
        '''

        idx = 0
        for group in self._optim.param_groups:
            for p in group['params']:
                p.grad = grads[idx]
                idx += 1
        return

    def _pack_grad(self, objectives, ddp):

        # 输出:
        # - grad: 参数梯度的列表
        # - shape: 参数形状的列表
        # - has_grad: 表示参数是否具有梯度的掩码列表

        grads, shapes, has_grads = [], [], []  # 初始化梯度、形状和是否有梯度的列表
        for ii, obj in enumerate(objectives):  # 对每个目标函数进行迭代
            self._optim.zero_grad(set_to_none=True)  # 清零优化器的梯度
            if ii < len(objectives) - 1:  # 如果不是最后一个目标函数
                obj.backward(retain_graph=True)  # 反向传播，并保留计算图以备后续使用
            else:
                obj.backward(retain_graph=False)  # 反向传播，不保留计算图
            grad, shape, has_grad = self._retrieve_grad()  # 获取梯度、形状和是否有梯度
            # print("grad", len(grad))
            # print("grad===", grad[0].shape)
            # print("grad===", grad[10].shape)
            # print("grad===", grad[50].shape)
            # print("grad===", grad[100].shape)
            # print("shape", len(shape))
            # print("shape===", shape[0])
            grads.append(self._flatten_grad(grad, shape))  # 将梯度扁平化并添加到梯度列表中
            has_grads.append(self._flatten_grad(has_grad, shape))  # 将是否有梯度信息扁平化并添加到列表中
            shapes.append(shape)  # 添加形状到形状列表中
        return grads, shapes, has_grads  # 返回梯度、形状和是否有梯度的列表


    def _pack_grad1(self, objectives, ddp):
        '''
        pack the gradient of the parameters of the network for each objective
        
        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grads, shapes, has_grads = [], [], []
        for ii, obj in enumerate(objectives):
            self._optim.zero_grad(set_to_none=True)
            # if ii == 0: continue
            # out_tensors = list(_find_tensors(obj))
            # ddp.reducer.prepare_for_backward(out_tensors)
            if ii < len(objectives) - 1:
                obj.backward(retain_graph=True)
            else:
                obj.backward(retain_graph=False)
            grad, shape, has_grad = self._retrieve_grad()
            grads.append(self._flatten_grad(grad, shape))
            has_grads.append(self._flatten_grad(has_grad, shape))
            shapes.append(shape)
        return grads, shapes, has_grads


    def _unflatten_grad(self, grads, shapes):
        unflatten_grad, idx = [], 0
        # 遍历每个参数形状
        for shape in shapes:
            # 计算参数的长度（元素个数）
            length = np.prod(shape)
            # 从梯度列表中取出对应长度的梯度，并将其视图变形成参数的形状，然后克隆一份
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            # 更新索引，指向下一个参数的梯度
            idx += length
        return unflatten_grad


    def _unflatten_grad1(self, grads, shapes):
        unflatten_grad, idx = [], 0
        for shape in shapes:
            length = np.prod(shape)
            unflatten_grad.append(grads[idx:idx + length].view(shape).clone())
            idx += length
        return unflatten_grad

    def _flatten_grad(self, grads, shapes):
        # 将每个参数的梯度打平，并拼接成一个一维张量
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _flatten_grad1(self, grads, shapes):
        flatten_grad = torch.cat([g.flatten() for g in grads])
        return flatten_grad

    def _retrieve_grad(self):
        '''
        获取具有特定目标的网络参数的梯度信息

        输出：
        - grad：参数的梯度列表
        - shape：参数的形状列表
        - has_grad：表示参数是否有梯度的掩码列表
        '''

        grad, shape, has_grad = [], [], []
        # 遍历优化器的参数组
        for group in self._optim.param_groups:
            # 遍历参数组中的参数
            for p in group['params']:
                # 处理多头场景下梯度为空的情况
                if p.grad is None:
                    # 将参数的形状、梯度和掩码分别添加到对应的列表中
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                # 将参数的形状、梯度和掩码分别添加到对应的列表中
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad

    def _retrieve_grad1(self):
        '''
        get the gradient of the parameters of the network with specific
        objective

        output:
        - grad: a list of the gradient of the parameters
        - shape: a list of the shape of the parameters
        - has_grad: a list of mask represent whether the parameter has gradient
        '''

        grad, shape, has_grad = [], [], []
        for group in self._optim.param_groups:
            for p in group['params']:
                # if p.grad is None: continue
                # tackle the multi-head scenario
                if p.grad is None:
                    shape.append(p.shape)
                    grad.append(torch.zeros_like(p).to(p.device))
                    has_grad.append(torch.zeros_like(p).to(p.device))
                    continue
                shape.append(p.grad.shape)
                grad.append(p.grad.clone())
                has_grad.append(torch.ones_like(p).to(p.device))
        return grad, shape, has_grad


class TestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 4)

    def forward(self, x):
        return self._linear(x)


class MultiHeadTestNet(nn.Module):
    def __init__(self):
        super().__init__()
        self._linear = nn.Linear(3, 2)
        self._head1 = nn.Linear(2, 4)
        self._head2 = nn.Linear(2, 4)

    def forward(self, x):
        feat = self._linear(x)
        return self._head1(feat), self._head2(feat)


if __name__ == '__main__':

    # fully shared network test
    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = TestNet()
    y_pred = net(x)
    pc_adam = GMD(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.L1Loss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred, y), loss2_fn(y_pred, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)

    print('-' * 80)
    # seperated shared network test

    torch.manual_seed(4)
    x, y = torch.randn(2, 3), torch.randn(2, 4)
    net = MultiHeadTestNet()
    y_pred_1, y_pred_2 = net(x)
    pc_adam = GMD(optim.Adam(net.parameters()))
    pc_adam.zero_grad()
    loss1_fn, loss2_fn = nn.MSELoss(), nn.MSELoss()
    loss1, loss2 = loss1_fn(y_pred_1, y), loss2_fn(y_pred_2, y)

    pc_adam.pc_backward([loss1, loss2])
    for p in net.parameters():
        print(p.grad)
