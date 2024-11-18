import itertools
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

class SegmentationLosses(object):
    def __init__(self, args, nclass=3, ):
        self.args = args
        self.cuda = self.args.cuda
        self.nclass = nclass
        self.VID_branch = VIDLoss(num_input_channels=128, num_mid_channel=128,
                                  num_target_channels=128)

    def build_loss(self, mode='ce'):
        if mode == 'enumeration':
            print("mode == 'enumeration'")
            return self.EnumerationLoss
        else:
            print(f'Loss {mode} not available.')
            raise NotImplementedError
    

    def EnumerationLoss(self, logits, target, df, df_full, weights=None):
        M = len(logits)
        loss = 0.

        for l in reversed(range(1, 5)):
            for subset in itertools.combinations(list(range(M)), l):

                missing_logits = torch.stack([logits[l1] for l1 in subset], dim=0)
                df1 = torch.stack([df[l1] for l1 in subset], dim=0)

                missing_logits = torch.mean(missing_logits, dim=0)
                df2 = torch.mean(df1, dim=0)
                self.VID_branch = self.VID_branch.cuda()
                MI_pred_mean, MI_log_scale = self.VID_branch(df2)


                loss += self.DiceCoef(missing_logits, target) + self.Cal_MIloss(df_full, MI_pred_mean, MI_log_scale)

        if self.args.loss.output == 'mean':
            print("self.args.loss.output == 'mean'")
            loss /= len(list(itertools.combinations(list(range(M)), M - self.args.loss.missing_num)))

        return loss



    def DiceCoef(self, preds, targets):
        alpha = 1.1
        beta = alpha / (alpha - 1)
        smooth = 1.0
        class_num = self.nclass
        sigmoid = nn.Sigmoid()
        preds = sigmoid(preds)

        loss = torch.zeros(class_num, device=preds.device)
        for i in range(class_num):
            pred = preds[:, i, :, :]
            target = targets[:, i, :, :]
            intersection = (pred * target).sum()
            loss_dice = 1 - ((2.0 * intersection + smooth) / (pred.sum() + target.sum() + smooth))

            p = F.softmax(pred, dim=1)
            q = F.softmax(target, dim=1)

            holder = -torch.log((p * q).sum() / ((p ** alpha).sum() ** (1 / alpha) * (q ** beta).sum() ** (1 / beta)))
            absolute_loss = abs(holder)
            loss_holder = absolute_loss

            loss[i] = loss_dice + loss_holder

        return torch.mean(loss)


    def Cal_MIloss(self, target, pred_mean, log_scale, init_pred_var=5.0, eps=1e-5):
        pred_var = torch.log(1.0 + torch.exp(log_scale)) + eps
        pred_var = pred_var.view(1, -1, 1, 1, 1)

        neg_log_prob = 0.5 * ((pred_mean - target) ** 2 / pred_var + torch.log(pred_var))

        loss = torch.mean(neg_log_prob)
        return loss


class VIDLoss(nn.Module):
    """Variational Information Distillation for Knowledge Transfer (CVPR 2019),
    code from author: https://github.com/ssahn0215/variational-information-distillation"""

    def __init__(self,
                 num_input_channels,
                 num_mid_channel,
                 num_target_channels,
                 init_pred_var=5.0,
                 eps=1e-5):
        super(VIDLoss, self).__init__()

        def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv3d(
                in_channels, out_channels,
                kernel_size=1, padding=0,
                bias=False, stride=stride)

        self.regressor = nn.Sequential(
            conv1x1(num_input_channels, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_mid_channel),
            nn.ReLU(),
            conv1x1(num_mid_channel, num_target_channels),
        )
        self.log_scale = torch.nn.Parameter(
            np.log(np.exp(init_pred_var - eps) - 1.0) * torch.ones(num_target_channels)
        )
        self.eps = eps

    def forward(self, x):
        pred_mean = self.regressor(x)
        log_scale = self.log_scale

        return pred_mean, log_scale