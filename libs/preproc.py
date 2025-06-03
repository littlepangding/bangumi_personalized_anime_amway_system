import torch
import torch.nn as nn
from torch.utils.data import Dataset

from collections import Counter, namedtuple
import hashlib
import pickle


# UserFeat = namedtuple(
#     "UserFeat",
#     [
#         "lt_5",
#         "lt_6",
#         "lt_7",
#         "lt_8",
#         "lt_9",
#         "lt_10",
#         "gt_5",
#         "gt_6",
#         "gt_7",
#         "gt_8",
#         "gt_9",
#     ],
# )

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


class UserShowRatingDataset(Dataset):
    def __init__(
        self,
        ratings,
        user_weight=None,
        pad_zero=True,
    ):
        self.ratings = ratings
        self.pad_zero = pad_zero
        self.user_weight = user_weight

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
        return len(self.ratings)

    def __getitem__(self, idx):
        u, s, r = self.ratings[idx]
        label = [1 if r > v else 0 for v in [5, 6, 7, 8, 9]]
        return (
            Feat(user_feat=self.user_feats[u], show_id=s),
            [torch.tensor(v) for v in label],
            [
                (
                    torch.tensor(self.user_weight[u])
                    if self.user_weight
                    else torch.tensor(1.0)
                )
                for v in label
            ],
        )


def custom_collate_fn(batch):
    # labels = [[], [], [], [], []]
    # weights = [[], [], [], [], []]
    # gt_input_and_offsets = [
    #     [[], []],
    #     [[], []],
    #     [[], []],
    #     [[], []]
    # ]
    # for f, l, w in batch:
    #     for i in range(5):
    #         labels[i].append(l[i])
    #         weights[i].append(w[i])
    #     show_ids.append(f.show_id)

    # for i in range(5):
    #     labels[i] = torch.tensor(labels[i])
    #     weights[i] = torch.tensor(weights[i])
    # show_ids = torch.LongTensor(show_ids)
    batch = list(batch)
    show_ids = torch.LongTensor([f.show_id for f, _, _ in batch])

    gt_input_and_offsets = []
    num_gt = 6  # len(batch[0][0].user_feat[0])
    for i in range(num_gt):
        offsets = torch.LongTensor([0] + [len(f.user_feat[0][i]) for f, _, _ in batch[:-1]])
        offsets = offsets.cumsum(0)
        input_tensor = torch.cat([f.user_feat[0][i] for f, _, _ in batch])
        gt_input_and_offsets.append((input_tensor, offsets))

    lt_input_and_offsets = []
    num_gt = 5  # len(batch[0][0].user_feat[1])
    for i in range(num_gt):
        offsets = torch.LongTensor([0] + [len(f.user_feat[1][i]) for f, _, _ in batch[:-1]])
        offsets = offsets.cumsum(0)
        input_tensor = torch.cat([f.user_feat[1][i] for f, _, _ in batch])
        lt_input_and_offsets.append((input_tensor, offsets))

    # get labels and weights
    num_labels = 5  # len(batch[0][1])
    # print()
    
    weights = [torch.tensor([w[i] for _, _, w in batch]) for i in range(num_labels)]
    labels = [torch.tensor([l[i] for _, l, _ in batch]) for i in range(num_labels)]
    return lt_input_and_offsets, gt_input_and_offsets, show_ids, labels, weights
