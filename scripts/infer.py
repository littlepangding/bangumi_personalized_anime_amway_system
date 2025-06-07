import os
from libs.model import load_model
import torch
from libs.model import PaaSModel
from libs import bgm_api, preproc, exp, bases
from torch.utils.data import DataLoader
from collections import Counter
import argparse
import json
import requests
import time
import re
from random import randint
from copy import deepcopy

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
exp_path = ""
bgm_auth_id = ""
old_to_new = {}
new_to_old = {}
ids_to_rec_set = set()
ids_to_rec_final = set()
show_details = {}
serving_ckp_dir_path = ""
version = "v0"


def print_show_detail_old(show_id):
    show_detail = show_details[show_id]
    return show_detail["name_cn"] or show_detail["name"]


def print_show_detail_new(show_id):
    s = new_to_old[show_id]
    return print_show_detail_old(
        s,
    )


def load_model_from_idx(exp_idx, ckp_idx):
    config_path = os.path.join(exp_path, f"exp{exp_idx:04d}", "config.json")
    ckp_path = os.path.join(exp_path, f"exp{exp_idx:04d}", f"model_{ckp_idx:02d}.ckp")
    return load_model(config_path, ckp_path, device)


def infer_for_user(user, model):
    msg = []
    print(f"Querying user rating/watching history for user from bangumi.tv: {user}")
    all_shows = bgm_api.get_all_results_for_user(bgm_auth_id, user)
    print(f"User rating/watching history retrieved, num. of show: {len(all_shows)}")

    print(f"Preparing user feature for inference")
    show_ratings, watched_shows = preproc.prep_infer_for_user(
        all_shows, old_to_new, ids_to_rec_set
    )
    msg.append(
        f"用户ID:{user}, 提取数据:{len(all_shows)}, "
        f"提取评分:{len(show_ratings)}, 已看:{len(watched_shows)}"
    )
    if not show_ratings:
        msg.append("似乎无法提取评分数据，无法生成个性化推荐。")
        return False, False, msg
    shows_to_infer = sorted(list(ids_to_rec_final - watched_shows))
    dataset = preproc.UserShowRatingInferDataset(
        show_ratings=show_ratings,
        shows_to_infer=shows_to_infer,
    )
    loader = DataLoader(
        dataset,
        batch_size=len(ids_to_rec_set),
        collate_fn=preproc.custom_collate_fn,
        shuffle=False,
    )
    print(f"User feature for inference prepared")

    print(f"Start user show watch/rating probability inference")
    for iter, (
        lt_input_and_offsets,
        gt_input_and_offsets,
        show_ids,
        _,
        _,
    ) in enumerate(loader):
        logits = exp.infer_and_get_logits(
            device,
            model,
            lt_input_and_offsets,
            gt_input_and_offsets,
            show_ids,
        )
        break
    print(f"User show watch/rating probability inference finished")
    return [l.detach().cpu().numpy() for l in logits], shows_to_infer, msg


def rec_top_k_v1(
    logits, shows_to_infer, k, k_min, mode=2, thd=0.2, show_to_exclude=None
):
    """
    mode:
        1: rank by review prob
        2n: rank by the rating alone
        2n+1 rank by the aggr. prob

    """
    review_prob = list(logits[-1])
    # rate 10, 9, 8, 7, 6
    ratex_prob = list(
        zip(list(logits[-2]), list(logits[-3]), list(logits[-4]), list(logits[-5]))
    )
    show_results = [
        (shows_to_infer[i], pr, px)
        for i, (pr, px) in enumerate(zip(review_prob, ratex_prob))
    ]

    def _get_value(pr, px, mode):
        if mode == 1:
            return pr, px[0]
        elif mode % 2 == 0:
            idx = mode / 2 - 1
            return px[idx], px[idx]
        else:
            idx = mode // 2 - 1
            return pr * px[idx], px[idx]

    show_results = sorted(show_results, key=lambda x: -_get_value(x[1], x[2], mode)[0])

    msg = []
    rec_shows = []
    for i, (s, pr, px) in enumerate(show_results[:k]):
        if show_to_exclude is not None and s in show_to_exclude:
            continue
        val, px_val = _get_value(pr, px, mode)
        if len(rec_shows) >= k_min and val < thd:
            break

        rec_shows.append(s)
        msg.append(
            f"推荐{len(rec_shows)}: {print_show_detail_new(s)}; 观看概率: {pr:03f}, 好评概率: {px_val:03f} 综合概率:{val:03f}"
        )
        print(
            f"Review: {pr:03f},\tRank Value: {val:03f}\t RateX: {[f'{x:03f}' for x in px]}"
            f"\tRec: {print_show_detail_new(s)}\tid: {new_to_old[s]}"
        )

    return msg


def rec_top_k(logits, shows_to_infer, k, k_min, mode=2, thd=0.2, show_to_exclude=None):
    review_prob = list(logits[-1])
    rate10_prob = list(logits[-2])
    rate9_prob = list(logits[-3])
    show_results = [
        (shows_to_infer[i], pr, p10, p10 * pr, p9, p9 * pr)
        for i, (pr, p10, p9) in enumerate(zip(review_prob, rate10_prob, rate9_prob))
    ]

    show_results = sorted(show_results, key=lambda x: -x[mode])

    msg = []
    rec_shows = []
    for i, (s, pr, p10, pr10, p9, pr9) in enumerate(show_results[:k]):
        if show_to_exclude is not None and s in show_to_exclude:
            continue
        if len(rec_shows) >= k_min and show_results[i][mode] < thd:
            break
        rec_shows.append(s)
        msg.append(
            f"推荐{len(rec_shows)}: {print_show_detail_new(s)}; 观看概率: {pr:03f}, 好评概率: {p9:03f} 综合概率:{pr9:03f}"
        )
        print(
            f"Review: {pr:03f},\tRate10: {p10:03f},\tAgg.10: {pr10:03f},\tRate9: {p9:03f},\tAgg.9: {pr9:03f},"
            f"\tRec: {print_show_detail_new(s)}\tid: {new_to_old[s]}"
        )

    return msg


def prepare(exp_idx, ckp_idx, data_path, review_lo_thd):
    (
        out_user_train,
        out_user_val,
        out_user_eval,
        in_user_train,
        in_user_val,
        in_user_eval,
        new_to_old,
        old_to_new,
        max_id,
    ) = preproc.load_split_data(data_path)
    model = load_model_from_idx(exp_idx, ckp_idx)
    ids_to_rec = sorted(list(set(s for u, s, r in out_user_train)))
    ids_to_rec_set = set(ids_to_rec)

    cnt = Counter()
    for u, s, r in out_user_train:
        cnt[s] += 1

    ids_to_rec_final = set(list(s for s, c in cnt.items() if c > review_lo_thd))
    print(
        f"Based on num review lowerbound: {review_lo_thd}, "
        f"only recommend from the top {len(ids_to_rec_final)} shows."
    )

    return (model, ids_to_rec_set, ids_to_rec_final, new_to_old, old_to_new)


def get_checkpoint_file_from_idx(idx):
    return os.path.join(serving_ckp_dir_path, f"serving_ckp_{idx:04d}.json")


def load_checkpoint(idx):
    if os.path.exists(get_checkpoint_file_from_idx(idx)):
        with open(get_checkpoint_file_from_idx(idx), "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def save_checkpoint(data, idx):
    with open(get_checkpoint_file_from_idx(idx), "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def extract_actual_id(text: str) -> str:
    """
    Extracts ACTUAL_ID from a string of the format:
    id ACTUAL_ID [anything]

    Returns the ACTUAL_ID if matched, else returns an empty string.
    """
    text = text.lstrip()  # Strip leading space-like characters

    # Regex breakdown:
    # ^id\s+       → starts with 'id' followed by one or more space-like characters
    # ([a-zA-Z0-9]+) → capture the ACTUAL_ID (alphanumeric)
    match = re.match(r"^id\s+([a-zA-Z0-9_]+)", text)
    if match:
        return match.group(1)
    return ""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-path", default="exp/keep/")
    parser.add_argument("--data-path", default="data/processed_data.pkl")
    parser.add_argument("--show-meta-path", default="data/show_metadata.pkl")
    parser.add_argument("--exp-idx", type=int, default=1629)
    parser.add_argument("--ckp-idx", type=int, default=0)
    parser.add_argument("--review-thd", type=int, default=1000)
    parser.add_argument("--user", default="")
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--k-min", type=int, default=5)
    parser.add_argument("--mode", type=int, default=5)
    parser.add_argument("--prob-thd", type=float, default=0.2)
    parser.add_argument("--serving-ckp-idx", type=int)
    parser.add_argument("--serving-ckp-dir-path", default="data/")
    parser.add_argument("--version", default="v0")
    parser.add_argument("--test", action="store_true")
    args = parser.parse_args()

    version = args.version
    print(f"param\tversion:{version}\t rec version")
    k = args.k
    print(f"param\tk:{k}\t recommend at most top k")
    k_min = args.k_min
    print(f"param\tk_min:{k_min}\t recommend at least top k")
    mode = args.mode
    print(f"param\tmode:{mode}\t recommendation criteria")
    prob_thd = args.prob_thd
    print(f"param\tprob_thd:{prob_thd}\t recommendation criteria lower bound")

    bgm_auth_id = bases.get_value("bgm_auth_id")
    exp_path = args.exp_path

    show_details = bases.load_show_details(args.show_meta_path)

    (
        model,
        ids_to_rec_set,
        ids_to_rec_final,
        new_to_old,
        old_to_new,
    ) = prepare(
        args.exp_idx,
        args.ckp_idx,
        args.data_path,
        args.review_thd,
    )

    if args.user:
        logits, shows_to_infer, msg = infer_for_user(args.user, model)

        print(f"Top {k} reconmmendation:")
        msg.extend(
            rec_top_k_v1(logits, shows_to_infer, k, k_min, mode=mode, thd=prob_thd)
        )
        recs = "\n".join(msg)
        print(f"recs message:\n{recs}")

    else:
        serving_ckp_dir_path = args.serving_ckp_dir_path
        serving_logging_ckp_idx = args.serving_ckp_idx
        # load existing ckeckpoint if exists
        replied_comments = load_checkpoint(serving_logging_ckp_idx)
        if replied_comments:
            # incr idx when it is not empty
            serving_logging_ckp_idx += 1

        bvid = bases.get_value("bvid")
        print(f"bvid: {bvid}")
        SESSDATA = bases.get_value("SESSDATA")
        csrf_token = bases.get_value("csrf_token")
        cookie = f"SESSDATA={SESSDATA}; bili_jct={csrf_token}"
        print(f"cookie: {cookie}")
        headers = {
            "Cookie": cookie,
            "User-Agent": "Mozilla/5.0",
            "Referer": f"https://www.bilibili.com/video/{bvid}",
        }

        oid = bgm_api.get_oid(bvid, headers)
        print(f"oid: {oid}")

        iter = 0
        while True:
            print(f"Start iteration: {iter}")
            if iter > 0:
                i = randint(5, 10)
                print(f"Will sleep for {i} minutes")
                time.sleep(60 * i)
            iter += 1
            print("Getting all comments")
            try:
                comments = bgm_api.get_all_comments(bvid, headers)
                print(f"Successfully got {len(comments)} comments")
            except Exception as e:
                print(f"Getting comments failed, will wait and retry: {e}")
                continue
            for m in comments:
                rpid = str(m["rpid"])
                message = m["message"]
                id = extract_actual_id(message)
                if not id:
                    continue
                if rpid in replied_comments:
                    if version in replied_comments[rpid]:
                        # already replied with this version, skipping
                        continue
                    if replied_comments[rpid].get("failed", 0) > 0:
                        # has failed before, skip
                        continue
                else:
                    replied_comments[rpid] = deepcopy(m)
                try:
                    # get recs TODO
                    logits, shows_to_infer, msg = infer_for_user(id, model)
                    if shows_to_infer:
                        msg.append("推荐如下")
                        msg.extend(
                            rec_top_k(
                                logits,
                                shows_to_infer,
                                k,
                                k_min,
                                mode=mode,
                                thd=prob_thd,
                            )
                        )
                    else:
                        pass
                    recs = "\n".join(msg)
                    print(f"recs message:\n{recs}")
                    assert (
                        bgm_api.reply(
                            oid, rpid, recs, headers, csrf_token, test_mode=args.test
                        )["code"]
                        == 0
                    )
                    replied_comments[rpid][version] = recs
                    print(f"Successfully replied to {rpid}")
                except Exception as e:
                    replied_comments[rpid]["failed"] = 1
                    print(f"failed to reply: {e}")
                print("Updating checkpoint...")
                save_checkpoint(replied_comments, serving_logging_ckp_idx)
                print("Checkpoint updated")
                time.sleep(randint(10, 20))
