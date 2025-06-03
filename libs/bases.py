import json
import pickle
from collections import Counter

KEY_FILE = "keys.json"


def get_value(key):
    with open(KEY_FILE, "r") as f:
        data = json.load(f)
        return data.get(key, None)
    return None


def load_show_details(filename):
    print(f"loading show details from {filename}")
    with open(filename, "rb") as handle:
        data = pickle.load(handle)
    show_details = data["show_details"]
    print(f"loaded {len(show_details)} show details.")
    return show_details


def load_ratings(filename):
    print(f"loading ratings from {filename}")
    with open(filename, "rb") as handle:
        data = pickle.load(handle)
    ratings = data["data"]
    print(f"loaded {len(ratings)} ratings.")
    return ratings


def save_show_details(filename, show_details):
    print(f"Saving {len(show_details)} show details to {filename}")
    with open(filename, "wb") as handle:
        pickle.dump(
            {"show_details": show_details},
            handle,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved {len(show_details)} show details.")
    return


def get_show_ids_above_num_ratings_thd(all_ratings, lo_thd):
    counter = Counter()
    for _, k, _ in all_ratings:
        counter[k] += 1
    return set(k for k, v in counter.items() if v >= lo_thd)
