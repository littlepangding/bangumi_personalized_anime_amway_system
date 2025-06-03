from libs import bases
import argparse
import os
from tqdm import tqdm
from libs import bgm_api

PAUSE = 0.5

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("to_filename")
    parser.add_argument("ratings_filename")
    parser.add_argument("--from_filename")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--num_rating_lo_thd", type=int, default=50)

    args = parser.parse_args()

    if args.from_filename == args.to_filename:
        if not args.force:
            raise ValueError(
                f"from and to filename has same name {args.from_filename}, using --force to override"
            )
    if not args.force and os.path.isfile(args.to_filename):
        raise ValueError(
            f"file {args.to_filename} already exists, to overide, use --force"
        )

    ratings = bases.load_ratings(args.ratings_filename)

    all_shows = set(s for _, s, _ in ratings)
    print(f"From ratings, get {len(all_shows)} unique shows.")

    all_shows = bases.get_show_ids_above_num_ratings_thd(
        ratings, args.num_rating_lo_thd
    )

    if args.from_filename:
        curr_show_details = bases.load_show_details(args.from_filename)
    else:
        curr_show_details = {}

    shows_to_pull = all_shows - set(k for k in curr_show_details)
    print(f"Still need to pull {len(shows_to_pull)} unique shows.")

    auth_id = bases.get_value("bgm_auth_id")
    for i, show_id in tqdm(enumerate(shows_to_pull)):
        try:
            val = bgm_api.get_show_detail_by_id(auth_id, show_id, PAUSE)
        except Exception as e:
            print(f"error met in {i} {show_id}.")
            continue
        curr_show_details[show_id] = val
        if i % 100 == 0:
            print(f"show :{show_id}\nresponse:{val}")

    bases.save_show_details(args.to_filename, curr_show_details)
