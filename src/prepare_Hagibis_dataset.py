from pathlib import Path
import json
import pickle
import unicodedata
import regex
import html
import pandas as pd
import geopandas as gpd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from shapely.geometry import Polygon, Point
from tqdm import tqdm
from twitter_api import TwitterAPI
mpl.rcParams['figure.dpi'] = 300


REGION_MAPPING_PATH = Path("data/japan_prefecture_district_mapping.csv")
SHAPEFILE_PATH = Path("data/japan_district_boundary/japan_ver83.shp")
RAW_TWEET_PATH = Path("data/typhoon_Hagibis_2019")
TWEET_OUTPUT_PATH = Path("data/typhoon_Hagibis_2019_dataset/typhoon_Hagibis_2019.pickle.zip")
PLACE_OUTPUT_PATH = Path("data/typhoon_Hagibis_2019_dataset/typhoon_Hagibis_2019_places.pickle")


def init_regions(path, level1_region="Prefecture", level2_region="District", level3_region="JCODE"):
    regions = dict()
    df = pd.read_csv(path)
    for _, row in df.iterrows():
        if row[level1_region] not in regions:
            regions[row[level1_region]] = dict()
        if row[level2_region] not in regions[row[level1_region]]:
            regions[row[level1_region]][row[level2_region]] = {"name": list(), "geometry": list()}
        regions[row[level1_region]][row[level2_region]]["name"].append(row[level3_region])
    return regions


def visualize_regions(map_df, regions, in_one=False, extension=1, cmap_name="Spectral"):
    if not in_one:
        for level1_region in regions:
            map_df.plot()
            xmin, xmax = 1e10, 0
            ymin, ymax = 1e10, 0
            for level2_region in regions[level1_region]:
                for shape in regions[level1_region][level2_region]["geometry"]:
                    if isinstance(shape, Polygon):
                        plt.fill(*shape.exterior.xy, fc='r')
                    else:
                        for geom in shape.geoms:
                            plt.fill(*geom.exterior.xy, fc='r')
                    xmin = min(xmin, shape.bounds[0])
                    xmax = max(xmax, shape.bounds[2])
                    ymin = min(ymin, shape.bounds[1])
                    ymax = max(ymax, shape.bounds[3])
            plt.xlim([xmin - extension, xmax + extension])
            plt.ylim([ymin - extension, ymax + extension])
            plt.show()
    else:
        map_df.plot()
        xmin, xmax = 1e10, 0
        ymin, ymax = 1e10, 0
        cmap = get_cmap(cmap_name)
        for rid, level1_region in enumerate(regions):
            color = cmap(rid / (len(regions) - 1))
            for level2_region in regions[level1_region]:
                for shape in regions[level1_region][level2_region]["geometry"]:
                    if isinstance(shape, Polygon):
                        plt.fill(*shape.exterior.xy, fc=color)
                    else:
                        for geom in shape.geoms:
                            plt.fill(*geom.exterior.xy, fc=color)
                    xmin = min(xmin, shape.bounds[0])
                    xmax = max(xmax, shape.bounds[2])
                    ymin = min(ymin, shape.bounds[1])
                    ymax = max(ymax, shape.bounds[3])
        plt.xlim([xmin - extension, xmax + extension])
        plt.ylim([ymin - extension, ymax + extension])
        plt.show()


def locate_point_in_regions(p, regions):
    tweet_location = None
    for level1_region in regions:
        for level2_region in regions[level1_region]:
            for shape in regions[level1_region][level2_region]["geometry"]:
                if shape.contains(p):
                    if tweet_location is None:
                        tweet_location = (level1_region, level2_region)
                    else:
                        raise RuntimeError(
                            "Multiple matches: {} ({}, {})".format(p, tweet_location, (level1_region, level2_region))
                        )
                    break
    if tweet_location is None:
        tweet_location = ("Unknown", "Unknown")

    return tweet_location


def locate_tweet(tweet, known_places, regions, twitter_api):
    tweet_loc = ("Unknown", "Unknown")
    place_id = "Unknown"

    if "geo" not in tweet:
        return tweet_loc, place_id

    if "place_id" in tweet["geo"]:
        place_id = tweet["geo"]["place_id"]

    if "coordinates" in tweet["geo"]:
        geo_type = tweet["geo"]["coordinates"]["type"]
        if geo_type == "Point":
            tweet_loc = locate_point_in_regions(
                Point(*tweet["geo"]["coordinates"]["coordinates"]), regions
            )
        else:
            raise RuntimeError("Unknown coordinate type: {}".format(geo_type))

    # Fallback to place centroid when no valid coordinate or cannot locate coordinate in regions
    if tweet_loc == ("Unknown", "Unknown"):
        if place_id not in known_places:
            place_info = twitter_api.get_place_information(place_id)
            if place_info is None:
                return tweet_loc, place_id  # Fail to locate the tweet

            # Assume that place centroid is within the district
            if "centroid" in place_info:
                tweet_loc = locate_point_in_regions(
                    Point(*place_info["centroid"]), regions
                )
            known_places[place_id] = (place_info, tweet_loc)
        else:
            _, tweet_loc = known_places[place_id]

    return tweet_loc, place_id


def clean_tweet_text(text):
    """ The text cleaning relies on https://github.com/hottolink/hottoSNS-bert """

    # charFilter: urlFilter
    url_regex = "https?://[-_.!~*\'()a-z0-9;/?:@&=+$,%#]+"
    url_pattern = regex.compile(url_regex, regex.IGNORECASE)

    # charFilter: partialurlFilter
    partial_url_regex = "(((https|http)(.{1,3})?)|(htt|ht))$"
    partial_url_pattern = regex.compile(partial_url_regex, regex.IGNORECASE)

    # charFilter: retweetflagFilter
    rt_regex = "rt (?=\@)"
    rt_pattern = regex.compile(rt_regex, regex.IGNORECASE)

    # charFilter: screennameFilter
    scname_regex = "\@[a-z0-9\_]+:?"
    scname_pattern = regex.compile(scname_regex, regex.IGNORECASE)

    # charFilter: truncationFilter
    truncation_regex = "…$"  # NFKC:"...$"
    truncation_pattern = regex.compile(truncation_regex, regex.IGNORECASE)

    # charFilter: hashtagFilter
    hashtag_regex = r"\#\S+"
    hashtag_pattern = regex.compile(hashtag_regex, regex.IGNORECASE)

    # charFilter: whitespaceNormalizer
    ws_regex = "\p{Zs}"
    ws_pattern = regex.compile(ws_regex, regex.IGNORECASE)

    # charFilter: controlcodeFilter
    cc_regex = "\p{Cc}"
    cc_pattern = regex.compile(cc_regex, regex.IGNORECASE)

    # charFilter: singlequestionFilter
    sq_regex = "\?{1,}"
    sq_pattern = regex.compile(sq_regex, regex.IGNORECASE)

    SPECIAL_TOKENS = {
        "url": "<url>",
        "screen_name": "<mention>"
    }

    # unescape html entities
    str_ = html.unescape(text)
    # charFilter: question mark
    str_ = sq_pattern.sub(" ", str_)
    # charFilter: strip
    str_ = str_.strip()
    # charFilter: truncationFilter
    str_ = truncation_pattern.sub("", str_)
    # charFilter: icuNormalizer(NKFC)
    str_ = unicodedata.normalize('NFKC', str_)
    # charFilter: caseNormalizer
    # str_ = str_.lower()
    # charFilter: retweetflagFilter
    str_ = rt_pattern.sub("", str_)
    # charFilter: partialurlFilter
    str_ = partial_url_pattern.sub("", str_)
    # charFilter: screennameFilter
    str_ = scname_pattern.sub("", str_)
    # charFilter: urlFilter
    str_ = url_pattern.sub("", str_)
    # charFilter: hastagFilter
    str_ = hashtag_pattern.sub("", str_)  # hashtags are removed for simplicity
    # charFilter: control code such as newline
    str_ = cc_pattern.sub(" ", str_)
    # charFilter: strip(once again)
    str_ = str_.strip()

    return str_


if __name__ == "__main__":
    kanto_regions = init_regions(REGION_MAPPING_PATH)

    # Retrieve districts from the shapefile
    japan_map = gpd.read_file(SHAPEFILE_PATH).set_index("JCODE")
    for prefecture in kanto_regions:
        for district in kanto_regions[prefecture]:
            for jcode in kanto_regions[prefecture][district]["name"]:
                jcode = '{:05d}'.format(jcode)
                region = japan_map.loc[jcode]
                if isinstance(region, pd.Series):
                    # JCODE should be unique in the map
                    kanto_regions[prefecture][district]["geometry"].append(region["geometry"])
    visualize_regions(japan_map, kanto_regions, in_one=True, extension=3)

    # Organize Raw Tweets into Districts
    tweets = dict()
    places = dict()
    places["Unknown"] = ({"id": "Unknown"}, ("Unknown", "Unknown"))  # Avoid errors when falling back to Unknown
    api = TwitterAPI()
    for loc in RAW_TWEET_PATH.iterdir():
        if loc.is_dir():
            for file in tqdm(sorted(loc.glob("*.json"))):
                data = json.load(open(str(file), "r"))
                for d in data["data"]:
                    if d["id"] not in tweets:  # Remove duplicates
                        tweet_loc, place_id = locate_tweet(d, places, kanto_regions, api)
                        tweets[d["id"]] = [
                            d["created_at"],
                            d["text"],
                            clean_tweet_text(d["text"]),
                            d["author_id"],
                            place_id,
                            tweet_loc[0],
                            tweet_loc[1],
                            d["conversation_id"],
                            d["public_metrics"]["retweet_count"],
                            d["public_metrics"]["reply_count"],
                            d["public_metrics"]["like_count"],
                            d["public_metrics"]["quote_count"],
                        ]

        # Make Checkpoint
        df = pd.DataFrame.from_dict(
            tweets, orient="index", columns=[
                "CreateTime", "Text", "CleanedText", "AuthorID", "PlaceID", "Prefecture", "District", "ConversationID",
                "RetweetCount", "ReplyCount", "LikeCount", "QuoteCount"
            ]
        )
        df.to_pickle(TWEET_OUTPUT_PATH)
        pickle.dump(places, open(str(PLACE_OUTPUT_PATH), "wb"))
