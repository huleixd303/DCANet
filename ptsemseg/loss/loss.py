import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F


def cross_entropy2d(input, target, weight=None, size_average=True):
    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        # target = target.unsequeeze(1)
        # target = F.upsample(target, size=(h, w), mode="nearest")
        # target = target.sequeeze(1)
        input = F.interpolate(input, size=(ht, wt), mode="biliner", align_corners=True)
    elif h < ht and w < wt:  # upsample images
        input = F.interpolate(input, size=(ht, wt), mode="bilinear", align_corners=True)
    elif h != ht and w != wt:
        raise Exception("Only support upsampling")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight = (torch.from_numpy(np.array(weight))).type(torch.Tensor)
    weight = weight.to(device)

    input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    target = target.view(-1)
    loss = F.cross_entropy(
        input, target, weight=weight, size_average=size_average, ignore_index=250
    )
    return loss


def multi_scale_cross_entropy2d(
    input, target, weight=None, size_average=True, scale_weight=None
):
    # Auxiliary training for PSPNet [1.0, 0.4] and ICNet [1.0, 0.4, 0.16]
    if scale_weight == None:  # scale_weight: torch tensor type
        n_inp = len(input)
        scale = 0.4
        scale_weight = torch.pow(scale * torch.ones(n_inp), torch.arange(n_inp))

    loss = 0.0
    for i, inp in enumerate(input):
        loss = loss + scale_weight[i] * cross_entropy2d(
            input=inp, target=target, weight=weight, size_average=size_average
        )

    return loss


def bootstrapped_cross_entropy2d(input,
                                  target, 
                                  K, 
                                  weight=None, 
                                  size_average=True):

    batch_size = input.size()[0]

    def _bootstrap_xentropy_single(input, 
                                   target, 
                                   K, 
                                   weight=None,
                                   size_average=True):

        n, c, h, w = input.size()
        input = input.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
        target = target.view(-1)
        loss = F.cross_entropy(input, 
                               target, 
                               weight=weight, 
                               reduce=False,
                               size_average=False, 
                               ignore_index=250)

        topk_loss, _ = loss.topk(K)
        reduced_topk_loss = topk_loss.sum() / K

        return reduced_topk_loss

    loss = 0.0
    # Bootstrap from each image not entire batch
    for i in range(batch_size):
        loss += _bootstrap_xentropy_single(
            input=torch.unsqueeze(input[i], 0),
            target=torch.unsqueeze(target[i], 0),
            K=K,
            weight=weight,
            size_average=size_average,
        )
    return loss / float(batch_size)

'''
class dice_bce_loss(nn.Module):
    def __init__(self, weight, batch=True):
        super(dice_bce_loss, self).__init__()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        weight = (torch.from_numpy(np.array(weight))).type(torch.Tensor)
        weight = weight.to(device)
        self.weight = weight
        self.batch = batch
        self.bce_loss = nn.BCELoss()

    def soft_dice_coeff(self, y_true, y_pred):
        smooth = 1.0  # may change
        if self.batch:
            i = torch.sum(y_true)
            j = torch.sum(y_pred)
            intersection = torch.sum(y_true * y_pred)
        else:
            i = y_true.sum(1).sum(1).sum(1)
            j = y_pred.sum(1).sum(1).sum(1)
            intersection = (y_true * y_pred).sum(1).sum(1).sum(1)
        score = (2. * intersection + smooth) / (i + j + smooth)
        # score = (intersection + smooth) / (i + j - intersection + smooth)#iou
        return score.mean()

    def soft_dice_loss(self, y_true, y_pred):
        loss = 1 - self.soft_dice_coeff(y_true, y_pred)
        return loss

    def __call__(self, y_pred, y_true):

        a = self.bce_loss(y_pred, y_true, self.weight)
        b = self.soft_dice_loss(y_true, y_pred)
        return a + b
'''
def MSE(input, target, size_average=True):
    target.float()
    input= input.squeeze(1)
    return F.mse_loss(input, target, size_average=True)


def dice_loss(input, target):
    # input = F.sigmoid(input)

    n, c, h, w = input.size()
    nt, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.float()
        target = target.unsqueeze(1)
        target = F.interpolate(target, size=(h, w), mode="nearest")
        target = target.squeeze(1)
    elif h < ht and w < wt:  # upsample images
        target = target.float()
        target = torch.unsqueeze(target, 1)
        target = F.interpolate(target, size=(h, w), mode="nearest")
        target = torch.squeeze(target, 1)


    smooth = 0.00001

    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # W_eight= (torch.from_numpy(np.array(W_eight))).type(torch.Tensor)
    # W_eight = W_eight.to(device)

    y_pred = input.view(-1)
    #print(y_pred.type())
    y_true = target.view(-1)
    #print(y_true.type())
    y_true= y_true.float()
    #print(y_true.type())


    i = torch.sum(y_true)
    j = torch.sum(y_pred)
    intersection = torch.sum(y_true * y_pred)
    score = (2. * intersection + smooth) / (i + j + smooth)
    soft_dice_coeff = score.mean()

    soft_dice_loss = 1 - soft_dice_coeff

    #a = bce_loss(input = y_pred, target = y_true, weight = W_eight, size_average =S_ize_average)
    #b = soft_dice_loss

    return soft_dice_loss

def bce_loss(input, target, weight =None, size_average = True):

    # input = F.sigmoid(input)
    _, _, h, w = input.size()
    _, ht, wt = target.size()

    # Handle inconsistent size between input and target
    if h > ht and w > wt:  # upsample labels
        target = target.float()
        target = target.unsequeeze(1)
        target = F.interpolate(target, size=(h, w), mode="nearest")
        target = target.sequeeze(1)
    elif h < ht and w < wt:  # upsample images
        target = target.float()
        target = torch.unsqueeze(target, 1)
        target = F.interpolate(target, size=(h, w), mode="nearest")
        target = torch.squeeze(target, 1)
        target = target.long()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weight =(torch.from_numpy(np.array(weight))).type(torch.Tensor)
    weight = weight.to(device)
    weight = weight.view(-1,1)
    target = target.contiguous().view(-1, 1)
    input = input.contiguous().view(-1,1)
    class_weight = torch.gather(weight, 0, target)
    target = target.float()
    # loss = F.binary_cross_entropy_with_logits(input, target, class_weight, size_average)
    loss = F.binary_cross_entropy(input, target, class_weight, size_average)
    return loss

def focal_loss(input, target, gamma = 2, alpha = None, size_average = True):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    input = input.contiguous().view(-1, 1)
    target = target.contiguous().view(-1, 1)

    pro = torch.cat((1-input, input), 1)
    #pro_0 = torch.cat((input, 1-input), 1)

    #target_float = target.float()
    #select_1 = (pro.gather(1,target))*(target_float) + 1e-9
    #select_0 = (pro.gather(1,1-target))*(target_float) + 1e-9

    select_init = torch.FloatTensor(len(pro), 2).zero_().to(device)
    select = select_init.scatter(1, target, 1.)

    if alpha is not None:
        weight = torch.tensor([[alpha], [1.0-alpha]])
        if weight.type() != input.data.type():
            weight = weight.type_as(input.data)
    weight = weight.to(device)

    weight = weight.view(-1,1)
    class_weight = torch.gather(weight, 0, target)


    pro_data = (pro*select).sum(1).view(-1, 1)
    pro_data = torch.clamp(pro_data, 1e-7,1-1e-7)
    batchloss = -class_weight*((1-pro_data)**gamma)*pro_data.log()

    # if alpha is not None:
    #     alpha = torch.tensor(alpha)
    #     if alpha.type() != input.data.type():
    #         alpha = alpha.type_as(input.data)
    # alpha = alpha.to(device)

    # pos_part = -1*(1-input)**gamma*(select_1.data.log())
    # p_sum = pos_part.sum()
    # neg_part = -1*input**gamma*(select_0.data.log())
    # n_sum = neg_part.sum()
    #
    # loss = alpha*pos_part + (1-alpha)*neg_part
    # p1_sum = (alpha*pos_part).sum()
    # n1_sum = ((1-alpha)*neg_part).sum()

    if size_average == True:
        loss = batchloss.mean()
    else:
        loss = batchloss

    return loss

# def dice_loss(y_pred, y_true):
#     smooth = 1.0
#
#     y_pred = y_pred.view(-1)
#     # print(y_pred.type())
#     y_true = y_true.view(-1)
#     # print(y_true.type())
#     y_true = y_true.float()
#     # print(y_true.type())
#
#     i = torch.sum(y_true)
#     j = torch.sum(y_pred)
#     intersection = torch.sum(y_true * y_pred)
#     score = (2. * intersection + smooth) / (i + j + smooth)
#     soft_dice_coeff = score.mean()
#
#     soft_dice_loss = 1 - soft_dice_coeff
#
#     return soft_dice_loss

def focal_dice_loss(input, target, gamma = 2, alpha = None, size_average = True):
    f_loss = focal_loss(input, target, gamma, alpha, size_average)
    d_loss = dice_loss(input, target)
    loss = f_loss + d_loss

    return loss

def dice_bce_loss(input, target, weight =None, size_average = True):
    bceloss = bce_loss(input, target, weight = weight, size_average = size_average)
    diceloss = dice_loss(input, target)
    loss = bceloss + diceloss
    return loss












