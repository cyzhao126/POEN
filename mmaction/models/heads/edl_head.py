import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import normal_init

from ..builder import HEADS
from .base import BaseHead


def get_subjective_info(alpha, evidence, classes):
    S = alpha.sum(dim=1, keepdim=True)
    b = evidence / S
    u = classes / S
    return b, u


def get_evidence_info(belief, uncertainty, classes):
    # calculate new S
    S_a = 2 / uncertainty
    # calculate new e_k
    e_a = torch.mul(belief, S_a)
    alpha_a = e_a + 1
    return alpha_a, e_a


def ds_combination_rule(b1, u1, b2, u2):
    b1b2 = torch.mul(b1, b2)
    b1u2 = torch.mul(b1, u2)
    b2u1 = torch.mul(b2, u1)
    u1u2 = torch.mul(u1, u2)

    C = (1 - (torch.mul(b1[:, 0], b2[:, 1]) + torch.mul(b2[:, 0], b1[:, 1]))).unsqueeze(1)

    combi_b = (b1b2 + b1u2 + b2u1) / C
    combi_u = u1u2 / C

    return combi_b, combi_u
@HEADS.register_module()
class EDLHead(BaseHead):
    """Classification head for I3D.

    Args:
        num_classes (int): Number of classes to be classified.
        in_channels (int): Number of channels in input feature.
        loss_cls (dict): Config for building loss.
            Default: dict(type='CrossEntropyLoss')
        spatial_type (str): Pooling type in spatial dimension. Default: 'avg'.
        dropout_ratio (float): Probability of dropout layer. Default: 0.5.
        init_std (float): Std value for Initiation. Default: 0.01.
        kwargs (dict, optional): Any keyword argument to be used to initialize
            the head.
    """

    def __init__(self,
                 num_classes,
                 in_channels,
                 loss_cls=dict(type='Eviloss_combination'),
                 spatial_type='avg',
                 dropout_ratio=0.5,
                 init_std=0.01,
                 **kwargs):
        super().__init__(num_classes, in_channels, loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes*2)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

    def init_weights(self):
        """Initiate the parameters from scratch."""
        normal_init(self.fc_cls, std=self.init_std)

    def forward(self, x):
        """Defines the computation performed at every call.

        Args:
            x (torch.Tensor): The input data.

        Returns:
            torch.Tensor: The classification scores for input samples.
        """
        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]

        ##### EDL

        # x = self.fc_cls(x)
        # alpha_list = []
        # first_evidence = F.softplus(x)
        # alpha_list.append(first_evidence + 1)
        #
        # # b1, u1 = get_subjective_info(alpha_list[0], first_evidence, self.num_classes)
        #
        # return alpha_list

        #### MSM
        x1 = x[::2]         #view1
        x2 = x[1::2]        #view2
        x1 = self.fc_cls(x1)
        x2 = self.fc_cls(x2)
        # [N, num_classes]
        alpha_list = []
        first_evidence = F.softplus(x1)
        alpha_list.append(first_evidence + 1)

        second_evidence = F.softplus(x2)
        alpha_list.append(second_evidence + 1)

        b1, u1 = get_subjective_info(alpha_list[0], first_evidence, self.num_classes)
        b2, u2 = get_subjective_info(alpha_list[1], second_evidence, self.num_classes)

        cmobi_b, combi_u = ds_combination_rule(b1, u1, b2, u2)

        combination_bu = (cmobi_b, combi_u)

        combi_alpha, _ = get_evidence_info(combination_bu[0], combination_bu[1], self.num_classes)

        alpha_list.append(combi_alpha)

        return alpha_list
