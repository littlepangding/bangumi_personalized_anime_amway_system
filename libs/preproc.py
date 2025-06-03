import torch
import torch.nn as nn
from collections import Counter


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


def remove_users_by_review_count(all_ratings, hi=None, lo=None):
    counter = Counter()
    for u, _, _ in all_ratings:
        counter[u] += 1

    users = {
        u: c
        for u, c in counter.items()
        if (c <= hi or hi is None) and (c >= lo or lo is None)
    }
    cleaned_ratings = [(u, s, r) for u, s, r in all_ratings if u in users]
    return cleaned_ratings, counter

def remove_shows_by_review_count(all_ratings, hi=None, lo=None):
    counter = Counter()
    for _, s, _ in all_ratings:
        counter[s] += 1
    shows = {
        s: c
        for u, c in counter.items()
        if (c <= hi or hi is None) and (c >= lo or lo is None)
    }
    cleaned_ratings = [(u, s, r) for u, s, r in all_ratings if u in shows]
    return cleaned_ratings, counter

def split_train_val_eval_data(all_ratings, ratio_inter, ratio_extra):
    """
    ratio is train:validate:eval data size;
        _inter is where train/validate/eval users has overlap
        _extra is where train/validate/eval users has no overlaps
        both will be generated, where _extra will be generated first
        and _inter will be generated from the train set of _extra
    """
    pass
