import torch
import torch.nn as nn
from torch.utils.data import Dataset

from collections import Counter, namedtuple
import hashlib
import pickle
from libs.model import HASH_SIZE, NUM_LABEL, NUM_GT, NUM_LT
from random import randint

Feat = namedtuple("Feat", ["user_feat", "show_id"])


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


def load_split_data(filename):
    print(f"loading splitted data from {filename}")
    with open(filename, "rb") as handle:
        data = pickle.load(handle)
    out_user_train = data["out_user_train"]
    out_user_val = data["out_user_val"]
    out_user_eval = data["out_user_eval"]
    in_user_train = data["in_user_train"]
    in_user_val = data["in_user_val"]
    in_user_eval = data["in_user_eval"]
    new_to_old = data["new_to_old"]
    old_to_new = data["old_to_new"]
    max_id = data["max_id"]

    print("data sizes after split:")
    print(f"\t out_user_train: {len(out_user_train)}")
    print(f"\t out_user_val: {len(out_user_val)}")
    print(f"\t out_user_eval: {len(out_user_eval)}")
    print(f"\t in_user_train: {len(in_user_train)}")
    print(f"\t in_user_val: {len(in_user_val)}")
    print(f"\t in_user_eval: {len(in_user_eval)}")
    print(f"\t max id: {max_id}")
    return (
        out_user_train,
        out_user_val,
        out_user_eval,
        in_user_train,
        in_user_val,
        in_user_eval,
        new_to_old,
        old_to_new,
        max_id,
    )


def prep_infer_for_user(all_shows, old_to_new, ids_to_rec_set):
    show_ratings = []
    watched_shows = set()
    for s, r, t, c in all_shows:
        if s not in old_to_new:
            continue
        n_s = old_to_new[s]
        if n_s not in ids_to_rec_set:
            continue
        if r > 0:
            show_ratings.append((n_s, r))
        if t == 1:
            pass
        else:
            watched_shows.add(n_s)
    return show_ratings, watched_shows


class UserShowRatingDataset(Dataset):
    def __init__(
        self,
        ratings,
        # user_weight=None,
        pad_zero=True,
        neg_sampling=True,
        user_id_exclude_paired_show=False,
    ):
        self.ratings = ratings
        self.pad_zero = pad_zero
        # self.user_weight = user_weight
        self.neg_sampling = neg_sampling
        self.user_id_exclude_paired_show = user_id_exclude_paired_show

        # get all users and their corresponding shows ids
        self.user_feats = self._prepare_users()

    def _prepare_users(self):
        user_to_show_ratings = {}
        for u, s, r in self.ratings:
            if u not in user_to_show_ratings:
                user_to_show_ratings[u] = []
            user_to_show_ratings[u].append((s, r))

        user_feats = {}
        for u, s_r in user_to_show_ratings.items():
            uf = [
                # lt ids
                [
                    torch.LongTensor(
                        [s for s, r in s_r if r < 5] + [0] if self.pad_zero else []
                    ),
                    torch.LongTensor(
                        [s for s, r in s_r if r < 6] + [0] if self.pad_zero else []
                    ),
                    torch.LongTensor(
                        [s for s, r in s_r if r < 7] + [0] if self.pad_zero else []
                    ),
                    torch.LongTensor(
                        [s for s, r in s_r if r < 8] + [0] if self.pad_zero else []
                    ),
                    torch.LongTensor(
                        [s for s, r in s_r if r < 9] + [0] if self.pad_zero else []
                    ),
                    torch.LongTensor(
                        [s for s, r in s_r if r < 10] + [0] if self.pad_zero else []
                    ),
                ],
                # gt ids
                [
                    torch.LongTensor(
                        [s for s, r in s_r if r > 5] + [0] if self.pad_zero else []
                    ),
                    torch.LongTensor(
                        [s for s, r in s_r if r > 6] + [0] if self.pad_zero else []
                    ),
                    torch.LongTensor(
                        [s for s, r in s_r if r > 7] + [0] if self.pad_zero else []
                    ),
                    torch.LongTensor(
                        [s for s, r in s_r if r > 8] + [0] if self.pad_zero else []
                    ),
                    torch.LongTensor(
                        [s for s, r in s_r if r > 9] + [0] if self.pad_zero else []
                    ),
                ],
            ]
            user_feats[u] = uf
        return user_feats

    # def _prepare_user_weights(self):
    #     self.user_review_count = Counter()
    #     for u, _, _ in self.ratings:
    #         self.user_review_count[u] += 1.0

    def __len__(self):
        return len(self.ratings) if not self.neg_sampling else 2 * len(self.ratings)

    def __getitem__(self, idx):
        neg_sample = idx >= len(self.ratings)
        if neg_sample:
            u, _, r = self.ratings[idx - len(self.ratings)]
            s = randint(1, HASH_SIZE - 1)
        else:
            u, s, r = self.ratings[idx]

        if not self.neg_sampling:
            label = [1.0 if r > v else 0.0 for v in [5, 6, 7, 8, 9]]
            weights = [torch.tensor(1.0)] * NUM_LABEL
        elif neg_sample:
            label = [0.0] * (NUM_LABEL - 1) + [0.0]
            weights = [0.0] * (NUM_LABEL - 1) + [1.0]
        else:
            label = [1.0 if r > v else 0.0 for v in [5, 6, 7, 8, 9]] + [1.0]
            weights = [1.0] * (NUM_LABEL - 1) + [1.0]

        user_feats = self.user_feats[u]
        if self.user_id_exclude_paired_show:
            user_feats_removed = [
                [v[v != s] for v in user_feats[0]],
                [v[v != s] for v in user_feats[1]],
            ]
        return (
            Feat(user_feat=user_feats, show_id=s),
            label,
            weights,
        )


def custom_collate_fn(batch):
    batch = list(batch)
    show_ids = torch.LongTensor([f.show_id for f, _, _ in batch])

    lt_input_and_offsets = []
    for i in range(NUM_LT):
        offsets = torch.LongTensor(
            [0] + [len(f.user_feat[0][i]) for f, _, _ in batch[:-1]]
        )
        offsets = offsets.cumsum(0)
        input_tensor = torch.cat([f.user_feat[0][i] for f, _, _ in batch])
        lt_input_and_offsets.append((input_tensor, offsets))

    gt_input_and_offsets = []
    for i in range(NUM_GT):
        offsets = torch.LongTensor(
            [0] + [len(f.user_feat[1][i]) for f, _, _ in batch[:-1]]
        )
        offsets = offsets.cumsum(0)
        input_tensor = torch.cat([f.user_feat[1][i] for f, _, _ in batch])
        gt_input_and_offsets.append((input_tensor, offsets))

    # get labels and weights
    weights = [torch.tensor([w[i] for _, _, w in batch]) for i in range(NUM_LABEL)]
    labels = [torch.tensor([l[i] for _, l, _ in batch]) for i in range(NUM_LABEL)]
    return lt_input_and_offsets, gt_input_and_offsets, show_ids, labels, weights


class UserShowRatingInferDataset(Dataset):
    def __init__(
        self,
        show_ratings,
        shows_to_infer,
        pad_zero=True,
    ):
        self.show_ratings = show_ratings
        self.pad_zero = pad_zero
        self.shows_to_infer = shows_to_infer

        # get all users and their corresponding shows ids
        self.user_feat = self._prepare_users()

    def _prepare_users(self):
        s_r = self.show_ratings
        user_feats = [
            # lt ids
            [
                torch.LongTensor(
                    [s for s, r in s_r if r < 5] + [0] if self.pad_zero else []
                ),
                torch.LongTensor(
                    [s for s, r in s_r if r < 6] + [0] if self.pad_zero else []
                ),
                torch.LongTensor(
                    [s for s, r in s_r if r < 7] + [0] if self.pad_zero else []
                ),
                torch.LongTensor(
                    [s for s, r in s_r if r < 8] + [0] if self.pad_zero else []
                ),
                torch.LongTensor(
                    [s for s, r in s_r if r < 9] + [0] if self.pad_zero else []
                ),
                torch.LongTensor(
                    [s for s, r in s_r if r < 10] + [0] if self.pad_zero else []
                ),
            ],
            # gt ids
            [
                torch.LongTensor(
                    [s for s, r in s_r if r > 5] + [0] if self.pad_zero else []
                ),
                torch.LongTensor(
                    [s for s, r in s_r if r > 6] + [0] if self.pad_zero else []
                ),
                torch.LongTensor(
                    [s for s, r in s_r if r > 7] + [0] if self.pad_zero else []
                ),
                torch.LongTensor(
                    [s for s, r in s_r if r > 8] + [0] if self.pad_zero else []
                ),
                torch.LongTensor(
                    [s for s, r in s_r if r > 9] + [0] if self.pad_zero else []
                ),
            ],
        ]
        return user_feats

    def __len__(self):
        return len(self.shows_to_infer)

    def __getitem__(self, idx):
        s = self.shows_to_infer[idx]
        return (
            Feat(user_feat=self.user_feat, show_id=s),
            [1] * NUM_LABEL,
            [torch.tensor(1.0)] * NUM_LABEL,
        )


def to_device(
    device,
    lt_input_and_offsets,
    gt_input_and_offsets,
    show_ids,
    labels=None,
    weights=None,
):

    return (
        [(i.to(device), o.to(device)) for i, o in lt_input_and_offsets],
        [(i.to(device), o.to(device)) for i, o in gt_input_and_offsets],
        show_ids.to(device) if show_ids is not None else None,
        [l.to(device) for l in labels] if labels is not None else None,
        [w.to(device) for w in weights] if weights is not None else None,
    )
