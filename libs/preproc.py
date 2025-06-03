import torch
import torch.nn as nn
from collections import Counter, namedtuple
import hashlib

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
        if (hi is None or c <= hi) and (lo is None or c >= lo)
    }
    cleaned_ratings = [(u, s, r) for u, s, r in all_ratings if u in users]
    return cleaned_ratings, counter


def remove_shows_by_review_count(all_ratings, hi=None, lo=None):
    counter = Counter()
    for _, s, _ in all_ratings:
        counter[s] += 1
    shows = {
        s: c
        for s, c in counter.items()
        if (hi is None or c <= hi) and (lo is None or c >= lo)
    }
    cleaned_ratings = [(u, s, r) for u, s, r in all_ratings if s in shows]
    return cleaned_ratings, counter


def compress_show_ids(all_ratings, start_idx=1):
    old_to_new = {}
    new_to_old = {}
    for _, s, _ in all_ratings:
        if s in old_to_new:
            continue
        new_id = len(old_to_new) + start_idx
        old_to_new[s] = new_id
        new_to_old[new_id] = s
    max_id = new_id + 1
    compressed_ratings = [(u, old_to_new[s], r) for u, s, r in all_ratings]
    return compressed_ratings, old_to_new, new_to_old, max_id


def string_to_int(s, max_int):
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    val = int(h, 16)
    if max_int:
        val = val % max_int
    return val


def split_train_val_eval_data(all_ratings, in_user_ratio, out_user_ratio):
    """
    ratio is train:validate:eval data size;
        in_user is where train/validate/eval users has overlap
        out_user is where train/validate/eval users has no overlaps
        both will be generated, where out_user will be generated first
        and in_user will be generated from the train set of out_user
    """

    out_user_split = [int(v) for v in out_user_ratio.split(":")]
    in_user_split = [int(v) for v in in_user_ratio.split(":")]
    assert len(out_user_split) == 3
    assert len(in_user_split) == 3
    for v in out_user_split:
        assert v >= 0, out_user_split
    for v in in_user_split:
        assert v >= 0, in_user_split
    # get out user split
    out_user_split = [int(v) for v in out_user_ratio.split(":")]
    max_out_user = sum(out_user_split)
    out_user_train = [
        (u, s, r)
        for u, s, r in all_ratings
        if string_to_int(str(u), max_out_user) <= out_user_split[0]
    ]
    out_user_val = [
        (u, s, r)
        for u, s, r in all_ratings
        if string_to_int(str(u), max_out_user) > out_user_split[0]
        and string_to_int(str(u), max_out_user) <= out_user_split[0] + out_user_split[1]
    ]
    out_user_eval = [
        (u, s, r)
        for u, s, r in all_ratings
        if string_to_int(str(u), max_out_user) > out_user_split[0] + out_user_split[1]
    ]

    # get in user split
    in_user_split = [int(v) for v in in_user_ratio.split(":")]
    max_in_user = sum(in_user_split)

    in_user_train = [
        (u, s, r)
        for u, s, r in out_user_train
        if string_to_int(str(u) + str(s), max_in_user) <= in_user_split[0]
    ]
    in_user_val = [
        (u, s, r)
        for u, s, r in out_user_train
        if string_to_int(str(u) + str(s), max_in_user) > in_user_split[0]
        and string_to_int(str(u) + str(s), max_in_user)
        <= in_user_split[0] + in_user_split[1]
    ]
    in_user_eval = [
        (u, s, r)
        for u, s, r in out_user_train
        if string_to_int(str(u) + str(s), max_in_user)
        > in_user_split[0] + in_user_split[1]
    ]

    return (
        out_user_train,
        out_user_val,
        out_user_eval,
        in_user_train,
        in_user_val,
        in_user_eval,
    )
