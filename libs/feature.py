import torch
import torch.nn as nn


from collections import namedtuple

UserFeat = namedtuple(
    "UserFeat",
    [
        "lt_5",
        "lt_6",
        "lt_7",
        "lt_8",
        "lt_9",
        "lt_10",
        "gt_5",
        "gt_6",
        "gt_7",
        "gt_8",
        "gt_9",
    ],
)
