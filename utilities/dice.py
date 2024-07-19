'''
Code adapted from: https://github.com/shubhamMehla12/torchgeometry_d/blob/main/torchgeometry_d/losses/dice.py
'''
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# based on:
# https://github.com/kevinzakka/pytorch-goodies/blob/master/losses.py


def one_hot(labels: torch.Tensor,
            num_classes: int,
            device: Optional[torch.device] = None,
            dtype: Optional[torch.dtype] = None,
            eps: Optional[float] = 1e-6) -> torch.Tensor:
    r"""Converts an integer label 2D tensor to a one-hot 3D tensor.
    Args:
        labels (torch.Tensor) : tensor with labels of shape :math:`(N, H, W)`,
                                where N is batch siz. Each value is an integer
                                representing correct classification.
        num_classes (int): number of classes in labels.
        device (Optional[torch.device]): the desired device of returned tensor.
         Default: if None, uses the current device for the default tensor type
         (see torch.set_default_tensor_type()). device will be the CPU for CPU
         tensor types and the current CUDA device for CUDA tensor types.
        dtype (Optional[torch.dtype]): the desired data type of returned
         tensor. Default: if None, infers data type from values.
    Returns:
        torch.Tensor: the labels in one hot tensor.
    Examples::
        >>> labels = torch.LongTensor([[[0, 1], [2, 0]]])
        >>> tgm.losses.one_hot(labels, num_classes=3)
        tensor([[[[1., 0.],
                  [0., 1.]],
                 [[0., 1.],
                  [0., 0.]],
                 [[0., 0.],
                  [1., 0.]]]]
    """
    if not torch.is_tensor(labels):
        raise TypeError("Input labels type is not a torch.Tensor. Got {}"
                        .format(type(labels)))
    if not len(labels.shape) == 3:
        raise ValueError("Invalid depth shape, we expect BxHxW. Got: {}"
                         .format(labels.shape))
    if not labels.dtype == torch.int64:
        raise ValueError(
            "labels must be of the same dtype torch.int64. Got: {}" .format(
                labels.dtype))
    if num_classes < 1:
        raise ValueError("The number of classes must be bigger than one."
                         " Got: {}".format(num_classes))
    batch_size, height, width = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, height, width,
                          device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0) + eps


class DiceLoss(nn.Module):
    r"""Criterion that computes Sørensen-Dice Coefficient loss.
    According to [1], we compute the Sørensen-Dice Coefficient as follows:
    .. math::
        \text{Dice}(x, class) = \frac{2 |X| \cap |Y|}{|X| + |Y|}
    where:
       - :math:`X` expects to be the scores of each class.
       - :math:`Y` expects to be the one-hot tensor with the class labels.
    the loss, is finally computed as:
    .. math::
        \text{loss}(x, class) = 1 - \text{Dice}(x, class)
    [1] https://en.wikipedia.org/wiki/S%C3%B8rensen%E2%80%93Dice_coefficient
    Shape:
        - Input: :math:`(N, C, H, W)` where C = number of classes.
        - Target: :math:`(N, H, W)` where each value is
          :math:`0 ≤ targets[i] ≤ C−1`.
    Examples:
        >>> N = 5  # num_classes
        >>> loss = tgm.losses.DiceLoss()
        >>> pred = torch.randn(1, N, 3, 5, requires_grad=True)
        >>> target = torch.empty(1, 3, 5, dtype=torch.long).random_(N)
        >>> output = loss(pred, target)
        >>> output.backward()
    """

    def __init__(self, ignore_index=None, use_softmax=False) -> None:
        super(DiceLoss, self).__init__()
        self.eps = 1e-6
        self.ignore_index = ignore_index
        self.use_softmax = use_softmax

    def forward(
            self,
            pred: torch.Tensor,
            target: torch.Tensor) -> torch.Tensor:
        if not torch.is_tensor(pred):
            raise TypeError("Input type is not a torch.Tensor. Got {}"
                            .format(type(pred)))
        if not len(pred.shape) == 4:
            raise ValueError("Invalid input shape, we expect BxNxHxW. Got: {}"
                             .format(pred.shape))
        if not pred.shape[-2:] == target.shape[-2:]:
            raise ValueError("input and target shapes must be the same. Got: {}"
                             .format(pred.shape, pred.shape))
        if not pred.device == target.device:
            raise ValueError(
                "input and target must be in the same device. Got: {}" .format(
                    pred.device, target.device))

        target_cl = target.clone()
        pred_cl = pred.clone()

        # Remove index to be ignored
        if self.ignore_index is not None:
            mask = (target_cl != self.ignore_index)
        else:
            mask = torch.ones(*target_cl.shape)
        target_cl = target_cl * mask

        # # compute softmax over the classes axis
        # input_soft = F.softmax(pred, dim=1)

        # create the labels one hot tensor
        target_one_hot = one_hot(target_cl, num_classes=pred.shape[1],
                                 device=pred.device, dtype=target_cl.dtype)

        if self.use_softmax:
            pred_cl = F.softmax(pred_cl, dim=1)

        # compute the actual dice score
        dims = (1, 2, 3)
        intersection = torch.sum(pred_cl * target_one_hot, dims)
        cardinality = torch.sum(pred_cl + target_one_hot, dims)

        dice_score = 2. * intersection / (cardinality + self.eps)
        return torch.mean(1. - dice_score)


######################
# functional interface
######################


def dice_loss(pred, target, ignore_index=None):
    r"""Function that computes Sørensen-Dice Coefficient loss.
    See :class:`~torchgeometry.losses.DiceLoss` for details.
    """
    return DiceLoss(ignore_index=ignore_index)(pred, target)
