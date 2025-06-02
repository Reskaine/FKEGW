import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


hiera = {
    "hiera_high":{
        "complete melting": [0, 1],
        "partial melting": [1, 4],
        "no melting": [4, 5],
    }
}

def prepare_targets(targets):
    b = targets.shape
    targets_high = torch.zeros(b, dtype=targets.dtype, device=targets.device)
    indices_high = []
    for index, high in enumerate(hiera["hiera_high"].keys()):
        indices = hiera["hiera_high"][high]
        for ii in range(indices[0], indices[1]):
            targets_high[targets == ii] = index
        indices_high.append(indices)

    return targets, targets_high, indices_high


def losses_bce_focal(predictions, targets, eps=1e-8, alpha=0.5, gamma=2):
    predictions = torch.sigmoid(predictions.float())
    loss = ((-alpha*targets*torch.pow((1.0-predictions),gamma)*torch.log(predictions+eps)
             -(1-alpha)*(1.0-targets)*torch.pow(predictions, gamma)*torch.log(1.0-predictions+eps))
            ).sum()

    return loss


def loss_e(predictions, predictions_plus, num_classes, p=3.0):
    predictions = torch.sigmoid(predictions.float())
    predictions = torch.cat([predictions, predictions_plus], dim=1)

    MCMA = predictions[:, :-3]
    MCMB = predictions[:, -3:]
    # filter high confidence pixels
    easy_A_pos = (MCMA > 0.7).sum(-1)
    easy_A_neg = (MCMA < 0.3).sum(-1)
    hard_A = 1 - torch.logical_and(easy_A_pos == 1, easy_A_neg == num_classes - 1).float()

    easy_B_pos = (MCMB > 0.7).sum(-1)
    easy_B_neg = (MCMB < 0.3).sum(-1)
    hard_B = 1 - torch.logical_and(easy_B_pos == 1, easy_B_neg == 2).float()

    num_hard_A = hard_A.sum().item()
    num_hard_B = hard_B.sum().item()

    if num_hard_A == 0:
            all_A = torch.tensor(1e-7, device=predictions.device)
    else:
        new_MCMA = MCMA[hard_A > 0].unsqueeze(-1)  # num_hard, num_class, 1
        mask_A = (1 - torch.eye(num_classes))[None, :, :].cuda()
        predicate_A = (new_MCMA @ (new_MCMA.transpose(1, 2))) * mask_A
        all_A = torch.pow(torch.pow(predicate_A, p).mean(), 1.0 / p)

    if num_hard_B == 0:
        all_B = torch.tensor(1e-7, device=predictions.device)
    else:
        new_MCMB = MCMB[hard_B > 0].unsqueeze(-1)  # num_hard, 3, 1
        mask_B = (1 - torch.eye(3))[None, :, :].cuda()
        predicate_B = (new_MCMB @ (new_MCMB.transpose(1, 2))) * mask_B
        all_B = torch.pow(torch.pow(predicate_B, p).mean(), 1.0 / p)

    # 2. average the clauses
    factor_A = num_classes * num_classes / (num_classes * num_classes + 3 * 3)
    factor_B = 3 * 3 / (num_classes * num_classes + 3 * 3)
    loss_ex = all_A * factor_A + all_B * factor_B

    return loss_ex


def loss_c(predictions, predictions_plus, num_classes, indices_high, eps=1e-8, p=5):
    predictions = torch.sigmoid(predictions.float())
    predictions = torch.cat([predictions, predictions_plus], dim=1)

    MCMA = predictions[:, :-3]
    MCMB = predictions[:, -3:]

    # predicate: 1-p+p*q, with aggregater, simplified to p-p*q
    predicate = MCMA.clone()
    for ii in range(3):
        indices = indices_high[ii]
        predicate[:, indices[0]:indices[1]] = MCMA[:, indices[0]:indices[1]] - MCMA[:, indices[0]:indices[1]] * MCMB[:,
                                                                                                                ii:ii + 1]

    # for all clause: use pmeanError to aggregate
    #     loss_c = torch.pow(torch.pow(predicate, p).mean(dim=0), 1.0/p).sum()/num_classes
    loss_c = torch.pow(torch.pow(predicate, p).mean(), 1.0 / p)

    return loss_c


def loss_d(predictions, predictions_plus, num_classes, indices_high, eps=1e-8, p=5):
    predictions = torch.sigmoid(predictions.float())
    predictions = torch.cat([predictions, predictions_plus], dim=1)

    MCMA = predictions[:, :-3]
    MCMB = predictions[:, -3:]

    # predicate:  1-p+p*q, with aggregater, simplified to p-p*q
    predicate = MCMB.clone()
    for ii in range(3):
        indices = indices_high[ii]
        predicate[:, ii:ii + 1] = MCMB[:, ii:ii + 1] - MCMB[:, ii:ii + 1] * \
                                  MCMA[:, indices[0]:indices[1]].max(dim=1, keepdim=True)[0]

    # for all clause: use pmeanError to aggregate
    #     loss_d = torch.pow(torch.pow(predicate, p).mean(dim=0), 1.0/p).sum()/21
    loss_d = torch.pow(torch.pow(predicate, p).mean(), 1.0 / p)

    return loss_d


class MyLogicalLoss(nn.Module):
    def __init__(self):
        super(MyLogicalLoss, self).__init__()

    def forward(self, output, labels, step):
        targets, targets_top, indices_top = prepare_targets(labels)

        targets = F.one_hot(targets, 5)
        targets_top_ = F.one_hot(targets_top, 3)
        # targets = torch.cat([targets, targets_top], dim=1)

        hiera_loss = losses_bce_focal(output, targets)
        c_rule = loss_c(output, targets_top_, 5, indices_top)
        d_rule = loss_d(output, targets_top_, 5, indices_top)
        e_rule = loss_e(output, targets_top_, 5)

        if step < 750:
            factor = 0
        elif step < 1500:
            factor = float(step - 750) / 750.0
        else:
            factor = 1.0

        return hiera_loss, c_rule, d_rule, e_rule, factor


class MyWeightLoss(nn.Module):
    def __init__(self):
        super(MyWeightLoss, self).__init__()

    def forward(self, output, labels, m, num_class):
        label_ = torch.zeros(m, num_class)
        for i, label in enumerate(labels):
            label_[i, label] = 1.
        label_ = label_.cuda()
        loss1 = - (torch.log(output + 1e-9) * label_)
        loss = loss1.sum() / m

        return loss
