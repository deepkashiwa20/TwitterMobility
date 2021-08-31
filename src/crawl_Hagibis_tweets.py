from twitter_api import TwitterAPI
from pathlib import Path
from tqdm import tqdm
import logging
import json

logging.basicConfig(level=logging.INFO)
api = TwitterAPI()

root = Path("data/typhoon_Hagibis_2019")

for loc in [
    ("Ibaraki", 36.730868, 140.657335, 10, "loc1"),  # 茨城県
    ("Ibaraki", 36.767239, 140.358273, 5, "loc2"),
    ("Ibaraki", 36.624362, 140.382822, 8, "loc3"),
    ("Ibaraki", 36.376150, 140.516005, 18, "loc4"),
    ("Ibaraki", 36.219211, 140.111135, 12, "loc5"),
    ("Ibaraki", 36.260091, 139.888498, 4, "loc6"),
    ("Ibaraki", 36.108188, 139.934600, 7, "loc7"),
    ("Ibaraki", 36.158782, 139.772248, 4, "loc8"),
    ("Ibaraki", 36.193761, 139.707521, 1, "loc9"),
    ("Ibaraki", 35.988227, 140.033561, 6, "loc10"),
    ("Ibaraki", 35.980015, 140.235856, 8, "loc11"),
    ("Ibaraki", 36.069761, 140.427201, 10, "loc12"),
    ("Ibaraki", 36.000139, 140.594831, 5, "loc13"),
    ("Ibaraki", 35.919370, 140.689971, 5, "loc14"),
    ("Tochigi", 36.934749, 139.989213, 14, "loc1"),  # 栃木県
    ("Tochigi", 36.781214, 139.671425, 16, "loc2"),
    ("Tochigi", 36.579274, 139.986813, 15, "loc3"),
    ("Tochigi", 36.466860, 139.692937, 14, "loc4"),
    ("Tochigi", 36.758648, 140.167223, 5, "loc5"),
    ("Tochigi", 36.328020, 139.482053, 4, "loc6"),
    ("Tochigi", 36.245985, 139.730498, 3, "loc7"),
    ("Gunma", 36.776406, 139.137244, 12, "loc1"),  # 群馬県
    ("Gunma", 36.612417, 138.922947, 13, "loc2"),
    ("Gunma", 36.500496, 139.231080, 10, "loc3"),
    ("Gunma", 36.564511, 138.669928, 11, "loc4"),
    ("Gunma", 36.399511, 138.945599, 9, "loc5"),
    ("Gunma", 36.239947, 138.831516, 11, "loc6"),
    ("Gunma", 36.355436, 139.165218, 5, "loc7"),
    ("Gunma", 36.280774, 139.049776, 4, "loc8"),
    ("Gunma", 36.317719, 139.310881, 5, "loc9"),
    ("Gunma", 36.258060, 139.425543, 3, "loc10"),
    ("Gunma", 36.237681, 139.519499, 3, "loc11"),
    ("Gunma", 36.233791, 139.607047, 3, "loc12"),
    ("Saitama", 36.161749, 139.187436, 7, "loc1"),  # 埼玉県
    ("Saitama", 36.177538, 139.352385, 5, "loc2"),
    ("Saitama", 36.010687, 139.119443, 9, "loc3"),
    ("Saitama", 35.960781, 138.901426, 8, "loc4"),
    ("Saitama", 36.153076, 139.617140, 5, "loc5"),
    ("Saitama", 36.038543, 139.696008, 5, "loc6"),
    ("Saitama", 35.979287, 139.757049, 4, "loc7"),
    ("Saitama", 35.884103, 139.806855, 5, "loc8"),
    ("Saitama", 35.840065, 139.715248, 3, "loc9"),
    ("Saitama", 35.822720, 139.854117, 2, "loc10"),
    ("Saitama", 35.976217, 139.498332, 15, "loc11"),
    ("Chiba", 36.012915, 139.839819, 2, "loc1"),  # 千葉県
    ("Chiba", 35.965828, 139.868488, 2, "loc2"),
    ("Chiba", 35.924693, 139.913118, 3, "loc3"),
    ("Chiba", 35.850668, 139.977473, 5, "loc4"),
    ("Chiba", 35.708718, 140.080316, 10, "loc5"),
    ("Chiba", 35.787937, 139.903177, 3, "loc6"),
    ("Chiba", 35.855786, 140.085767, 2, "loc7"),
    ("Chiba", 35.829218, 140.212579, 2, "loc8"),
    ("Chiba", 35.643566, 139.922791, 3, "loc9"),
    ("Chiba", 35.661104, 140.476721, 11, "loc10"),
    ("Chiba", 35.717560, 140.823285, 3, "loc11"),
    ("Chiba", 35.341390, 140.148449, 18, "loc12"),
    ("Chiba", 35.089648, 139.978449, 14, "loc13"),
    ("Tokyo", 35.836867, 139.029683, 4, "loc1"),  # 東京都
    ("Tokyo", 35.770915, 139.124291, 7, "loc2"),
    ("Tokyo", 35.807673, 139.273310, 2, "loc3"),
    ("Tokyo", 35.706260, 139.302149, 7, "loc4"),
    ("Tokyo", 35.677254, 139.416133, 6, "loc5"),
    ("Tokyo", 35.579592, 139.451151, 2, "loc6"),
    ("Tokyo", 35.535448, 139.468481, 1, "loc7"),
    ("Tokyo", 35.756988, 139.509328, 2, "loc8"),
    ("Tokyo", 35.696213, 139.607027, 5, "loc9"),
    ("Tokyo", 35.745260, 139.683396, 4, "loc10"),
    ("Tokyo", 35.658289, 139.753977, 7, "loc11"),
    ("Tokyo", 35.773679, 139.787422, 2, "loc12"),
    ("Tokyo", 35.755847, 139.833771, 3, "loc13"),
    ("Tokyo", 35.706238, 139.879776, 2, "loc14"),
    ("Tokyo", 35.563301, 139.716701, 2, "loc15"),
    ("Tokyo", 35.552824, 139.781076, 2, "loc16"),
    ("Kanagawa", 35.630971, 139.163717, 2, "loc1"),  # 神奈川県
    ("Kanagawa", 35.548832, 139.208749, 5, "loc2"),
    ("Kanagawa", 35.565071, 139.327725, 3, "loc3"),
    ("Kanagawa", 35.416822, 139.115607, 7, "loc4"),
    ("Kanagawa", 35.434244, 139.355211, 9, "loc5"),
    ("Kanagawa", 35.246718, 139.152796, 9, "loc6"),
    ("Kanagawa", 35.229319, 139.529427, 13, "loc7"),
    ("Kanagawa", 35.463449, 139.610675, 8, "loc8"),
    ("Kanagawa", 35.499233, 139.747426, 3, "loc9"),
    ("Kanagawa", 35.594711, 139.541572, 3, "loc10"),
    ("Kanagawa", 35.584663, 139.601685, 2, "loc11"),
    ("Kanagawa", 35.550105, 139.502335, 1, "loc12"),
    ("Kanagawa", 35.577382, 139.653826, 1, "loc13"),
]:
    opath = root / loc[0]
    if not opath.exists():
        opath.mkdir()
    api.full_archive_search(
        opath,
        start_time="2019-09-22T00:00:00Z",
        end_time="2019-10-27T00:00:00Z",
        geo_point=(loc[1], loc[2], loc[3]),
        expansions=TwitterAPI.EXPANSIONS,
        tweet_fields=TwitterAPI.TWEET_FIELDS,
        place_fields=TwitterAPI.PLACE_FIELDS,
        media_fields=TwitterAPI.MEDIA_FIELDS,
        has_geo=True,
        max_results=100,
        prefix=loc[4]
    )

all_users = set()
for loc in ["Ibaraki", "Tochigi", "Gunma", "Saitama", "Chiba", "Tokyo", "Kanagawa"]:
    path = root / loc
    for file in tqdm(sorted(path.glob("*.json"))):
        data = json.load(open(str(file), "r"))
        for u in data["includes"]["users"]:
            all_users.add(u["id"])
print("{} Users found".format(len(all_users)))
json.dump(list(all_users), open(str(root / "all_users.json"), "w"))
