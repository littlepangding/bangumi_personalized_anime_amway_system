import requests
from time import sleep
import json


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


def get_user_detail_by_id(auth_id, user, page, pause=0.0):
    url = f"https://api.bgm.tv/v0/users/{user}/collections?subject_type=2&limit=50&offset={page}"
    response = requests.get(
        url,
        headers={
            "Authorization": f"Bearer {auth_id}",
            "User-Agent": "personalized_anime_amway_system",
        },
    )
    if pause > 0:
        sleep(pause)
    return json.loads(response.content)


def get_all_results_for_user(auth_id, user):
    total = -1
    page = 0
    all_shows = []
    shows_checked = set()
    while True:
        results = get_user_detail_by_id(auth_id, user, page * 50)
        if results.get("title", None) == "Bad Request":
            break
        if total <= 0:
            total = results["total"]
        else:
            assert total == results["total"]
        for r in results["data"]:
            s = int(r["subject_id"])
            assert s not in shows_checked, page
            shows_checked.add(s)
            all_shows.append(
                (
                    s,
                    float(r["rate"]),
                    int(r["type"]),
                    r["comment"],
                )
            )
        page += 1

    return all_shows


def get_oid(bvid, headers):
    url = f"https://api.bilibili.com/x/web-interface/view?bvid={bvid}"
    resp = requests.get(url, headers=headers)
    resp.raise_for_status()
    return resp.json()["data"]["aid"]


def get_all_comments(bvid, headers):
    comments = []
    oid = get_oid(bvid, headers)
    page = 1
    while True:
        url = f"https://api.bilibili.com/x/v2/reply?type=1&oid={oid}&pn={page}&ps=20&sort=2"
        resp = requests.get(url, headers=headers)
        if resp.status_code != 200:
            print(f"Failed to fetch comments on page {page}")
            break
        data = resp.json().get("data", {})
        replies = data.get("replies", [])
        if not replies:
            break
        for r in replies:
            comments.append({"rpid": r["rpid"], "message": r["content"]["message"]})
        page += 1
        sleep(1)
    return comments


def reply(oid, rpid, text, headers, csrf_token, test_mode=True):

    url = "https://api.bilibili.com/x/v2/reply/add"
    data = {
        "oid": oid,
        "type": 1,
        "message": text,
        "plat": 1,
        "csrf": csrf_token,
        "root": rpid,
        "parent": rpid,
    }
    if test_mode:
        print(f"testing reply: \n{url}\n{data}")
        return {"code": 0}
    resp = requests.post(url, data=data, headers=headers)

    return resp.json()
