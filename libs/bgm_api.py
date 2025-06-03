import requests
from time import sleep


def get_show_detail_by_id(auth_id, subject_id, pause=0.0):
    url = f"https://api.bgm.tv/v0/subjects/{subject_id}"
    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {auth_id}",
            "User-Agent": "personalized_anime_amway_system",
        },
    )
    if pause > 0:
        sleep(pause)
    return response.json()
