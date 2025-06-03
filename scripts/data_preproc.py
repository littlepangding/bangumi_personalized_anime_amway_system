import argparse
import os
import json
from libs import bases, preproc
import pickle


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("in_data")
    parser.add_argument("config")
    parser.add_argument("out_data")
    parser.add_argument("--force", action="store_true")

    args = parser.parse_args()
    if not args.force and os.path.isfile(args.out_data):
        raise ValueError(
            f"file {args.out_data} already exists, to overide, use --force"
        )

    config = bases.get_config(args.config)
    all_ratings = bases.load_ratings(args.in_data)
    print(f"start compressiong show id to be starting near 0 and consecutive")
    (
        compressed_ratings,
        old_to_new,
        new_to_old,
        max_id,
    ) = preproc.compress_show_ids(all_ratings)
    print(f"compression finished, max id: {max_id}")

    hi = config["preproc"]["show_review_cnt_hi"]
    lo = config["preproc"]["show_review_cnt_lo"]
    print(f"pruning ratings by the num of review for each show, hi: {hi}, lo: {lo}")
    print(f"num of ratigns before: {len(compressed_ratings)}")
    ratings, _ = preproc.remove_shows_by_review_count(
        compressed_ratings,
        hi=hi,
        lo=lo,
    )
    print(f"num of ratigns after: {len(ratings)}")

    hi = config["preproc"]["user_review_cnt_hi"]
    lo = config["preproc"]["user_review_cnt_lo"]
    print(f"pruning ratings by the num of review for each user, hi: {hi}, lo: {lo}")
    print(f"num of ratigns before: {len(ratings)}")
    ratings, _ = preproc.remove_users_by_review_count(
        ratings,
        hi=hi,
        lo=lo,
    )
    print(f"num of ratigns after: {len(ratings)}")

    # train, validation, eval split
    in_user_ratio = config["preproc"]["in_user_split_ratio"]
    out_user_ratio = config["preproc"]["out_user_split_ratio"]
    print(
        f"Data split config:\nin_user_ratio: {in_user_ratio}\nout_user_ratio: {out_user_ratio}"
    )
    (
        out_user_train,
        out_user_val,
        out_user_eval,
        in_user_train,
        in_user_val,
        in_user_eval,
    ) = preproc.split_train_val_eval_data(
        ratings,
        in_user_ratio=in_user_ratio,
        out_user_ratio=out_user_ratio,
    )
    print("data sizes after split:")
    print(f"\t out_user_train: {len(out_user_train)}")
    print(f"\t out_user_val: {len(out_user_val)}")
    print(f"\t out_user_eval: {len(out_user_eval)}")
    print(f"\t in_user_train: {len(in_user_train)}")
    print(f"\t in_user_val: {len(in_user_val)}")
    print(f"\t in_user_eval: {len(in_user_eval)}")

    with open(args.out_data, "wb") as handle:
        pickle.dump(
            {
                "out_user_train": out_user_train,
                "out_user_val": out_user_val,
                "out_user_eval": out_user_eval,
                "in_user_train": in_user_train,
                "in_user_val": in_user_val,
                "in_user_eval": in_user_eval,
                "new_to_old": new_to_old,
                "old_to_new": old_to_new,
                "max_id": max_id,
            },
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
