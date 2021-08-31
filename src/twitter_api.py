from typing import Dict, Tuple, List
from pathlib import Path
import logging, os, json, time, datetime, requests


class TwitterAPI(object):
    """ Wrapper class to access Twitter Data via Official Twitter API v2

    This class is developed based on the the Github repo:
        Twitter-API-v2-sample-code (https://github.com/twitterdev/Twitter-API-v2-sample-code)

    References:
        https://developer.twitter.com/en/docs/twitter-api
    """

    ENDPOINT_URLS = {
        "full_archive": "https://api.twitter.com/2/tweets/search/all",
        "tweet_lookup": "https://api.twitter.com/2/tweets",
        "place_information": "https://api.twitter.com/1.1/geo/id/{place_id}.json"
    }
    RATE_LIMITS = {
        "full_archive": (1, 300),  # (request per second, requests per 15-minute window)
        "tweet_lookup": (1, 300),
        "place_information": (1, 75)
    }
    EXPANSIONS = [
        "attachments.poll_ids", "attachments.media_keys", "author_id", "entities.mentions.username", "geo.place_id",
        "in_reply_to_user_id", "referenced_tweets.id", "referenced_tweets.id.author_id"
    ]
    TWEET_FIELDS = [
        "attachments", "author_id", "context_annotations", "conversation_id", "created_at", "entities",
        "geo", "id", "in_reply_to_user_id", "lang", "public_metrics", "possibly_sensitive",
        "referenced_tweets", "reply_settings", "source", "text", "withheld"
    ]
    USER_FIELDS = [
        "created_at", "description", "entities", "id", "location", "name", "pinned_tweet_id",
        "profile_image_url", "protected", "public_metrics", "url", "username", "verified", "withheld"
    ]
    MEDIA_FIELDS = [
        "duration_ms", "height", "media_key", "preview_image_url", "type", "url", "width", "public_metrics"
    ]
    PLACE_FIELDS = [
        "contained_within", "country", "country_code", "full_name", "geo", "id", "name", "place_type"
    ]
    POLL_FIELDS = [
        "duration_minutes", "end_datetime", "id", "options", "voting_status"
    ]

    def __init__(self, bearer_token: str = None):
        if bearer_token is None:
            self.bearer_token = os.environ.get("BEARER_TOKEN")
        else:
            self.bearer_token = bearer_token

        self.request_count = {service: 0 for service in self.ENDPOINT_URLS}
        now = datetime.datetime.now()
        self.cycle_start_time = {service: now for service in self.ENDPOINT_URLS}

    @staticmethod
    def geocode_point_query(latitude: float, longitude: float, radius: int, unit: str = "mi"):
        return "point_radius:[{:.6f} {:.6f} {:d}{}]".format(longitude, latitude, radius, unit)

    @staticmethod
    def geocode_box_query(west_long: float, south_lat: float, east_long: float, north_lat: float):
        return "bounding_box:[{:.6f} {:.6f} {:.6f} {:.6f}]".format(west_long, south_lat, east_long, north_lat)

    @staticmethod
    def from_user_query(user_id: str):
        return "from:{}".format(user_id)

    def full_archive_search(
            self, output_path: Path, keywords: str = None, from_user_id: str = None, start_time: str = None, end_time: str = None,
            expansions: List[str] = None, tweet_fields: List[str] = None, user_fields: List[str] = None,
            media_fields: List[str] = None, place_fields: List[str] = None, poll_fields: List[str] = None,
            geo_box: Tuple[float, float, float, float] = None, geo_point: Tuple[float, float, int] = None,
            has_geo: bool = None, is_retweet: bool = None,
            max_results: int = 10, next_token: str = None, start_page: int = None, prefix: str = "page"
    ):
        """ Access tweets via the full-archive search endpoint

        Args:
            output_path (Path): path to store all the responses.
            keywords (str): keywords for matching Tweet contents.
            from_user_id (str): user ID who posted the matching Tweets.
            start_time (str): the earliest UTC time for matching Tweets, in format of "YYYY-MM-DDTHH:mm:ssZ".
            end_time (str): the latest UTC time for matching Tweets, in format of "YYYY-MM-DDTHH:mm:ssZ".
            expansions (List[str]): the additional data objects to be included in the responses.
            tweet_fields (List[str]): the tweet fields to be included in the responses.
            user_fields (List[str]): the user fields to be included in the responses.
            media_fields (List[str]): the media fields to be included in the responses.
            place_fields (List[str]): the place fields to be included in the responses.
            poll_fields (List[str]): the poll fields to be included in the responses.
            geo_box (Tuple[float, float, float, float]):
                west longitude, south latitude, east longitude, north latitude for matching Tweets.
            geo_point (Tuple[float, float, int]): latitude, longitude, and radius in miles for matching Tweets.
            has_geo (bool): returns Tweets with/without geotags. Use None to include both.
            is_retweet (bool): include retweet or not. Use None to include both.
            max_results (int): the maximum number of search results to be included per request (between 10 and 500).
            next_token (str): the next token for pagination in Twitter API
            start_page (int): the page number for the first responses
            prefix (str): the prefix string added to the file name

        References:
            https://developer.twitter.com/en/docs/twitter-api/tweets/search/api-reference/get-tweets-search-all
        """

        query = list()
        if keywords is not None:
            query.append(keywords)
        if from_user_id is not None:
            query.append(self.from_user_query(from_user_id))
        if geo_box is not None:
            query.append(self.geocode_box_query(geo_box[0], geo_box[1], geo_box[2], geo_box[3]))
        if geo_point is not None:
            query.append(self.geocode_point_query(geo_point[0], geo_point[1], geo_point[2]))
        if has_geo:
            query.append("has:geo")
        else:
            query.append("-has:geo")
        if is_retweet:
            query.append("is:retweet")
        else:
            query.append("-is:retweet")

        params = {
            "query": " ".join(query),
            "max_results": max_results,
        }

        if start_time is not None:
            params["start_time"] = start_time
        if end_time is not None:
            params["end_time"] = end_time

        if expansions is not None:
            for exp in expansions:
                if exp not in self.EXPANSIONS:
                    raise ValueError("Unknown expansion: {}".format(exp))
            params["expansions"] = ",".join(expansions)
        if tweet_fields is not None:
            for f in tweet_fields:
                if f not in self.TWEET_FIELDS:
                    raise ValueError("Unknown tweet field: {}".format(f))
            params["tweet.fields"] = ",".join(tweet_fields)
        if user_fields is not None:
            for f in user_fields:
                if f not in self.USER_FIELDS:
                    raise ValueError("Unknown user field: {}".format(f))
            params["user.fields"] = ",".join(user_fields)
        if media_fields is not None:
            for f in media_fields:
                if f not in self.MEDIA_FIELDS:
                    raise ValueError("Unknown media field: {}".format(f))
            params["media.fields"] = ",".join(media_fields)
        if place_fields is not None:
            for f in place_fields:
                if f not in self.PLACE_FIELDS:
                    raise ValueError("Unknown place field: {}".format(f))
            params["place.fields"] = ",".join(place_fields)
        if poll_fields is not None:
            for f in poll_fields:
                if f not in self.POLL_FIELDS:
                    raise ValueError("Unknown poll field: {}".format(f))
            params["poll.fields"] = ",".join(poll_fields)

        if next_token is not None:
            params["next_token"] = next_token

        page = 1 if start_page is None else start_page
        while True:
            if datetime.datetime.now() - self.cycle_start_time["full_archive"] > datetime.timedelta(minutes=15):
                self.request_count["full_archive"] = 0
                self.cycle_start_time["full_archive"] = datetime.datetime.now()

            if self.request_count["full_archive"] < self.RATE_LIMITS["full_archive"][1]:
                try:
                    json_response = self._connect_to_endpoint("full_archive", params)
                except RuntimeError as err:
                    if err.args[1] == 429:  # Rate limit exceeded
                        logging.warning("Reach \"full archive\" rate limit")
                        seconds_to_next_cycle = self.cycle_start_time["full_archive"] + datetime.timedelta(minutes=15) - datetime.datetime.now()
                        time.sleep(max(10., seconds_to_next_cycle.total_seconds()))
                        continue
                    elif err.args[1] == 503:  # Service Unavailable
                        logging.warning("Service Unavailable")
                        time.sleep(60)
                        continue
                    else:
                        raise err

                json.dump(json_response, open(str(output_path / "{}_{:06d}.json".format(prefix, page)), "w"))
                if "next_token" not in json_response["meta"]:
                    break
                else:
                    params["next_token"] = json_response["meta"]["next_token"]
                page += 1
                self.request_count["full_archive"] += 1
                time.sleep(self.RATE_LIMITS["full_archive"][0])
            else:
                time.sleep(10)

    def tweet_lookup(
            self, output_path: Path, tweet_ids: List[str], retrieved_tweets: Dict[str, Dict] = None,
            expansions: List[str] = None, tweet_fields: List[str] = None, user_fields: List[str] = None,
            media_fields: List[str] = None, place_fields: List[str] = None, poll_fields: List[str] = None
    ):
        """ Lookup tweets by ids

        Args:
            output_path (Path): path to store all the responses.
            tweet_ids (List[str]): the ids to be retrieved.
            retrieved_tweets (Dict[str, Dict]): a dictionary with tweet IDs as keys and retrieved tweets as values.
            expansions (List[str]): the additional data objects to be included in the responses.
            tweet_fields (List[str]): the tweet fields to be included in the responses.
            user_fields (List[str]): the user fields to be included in the responses.
            media_fields (List[str]): the media fields to be included in the responses.
            place_fields (List[str]): the place fields to be included in the responses.
            poll_fields (List[str]): the poll fields to be included in the responses.
        """

        params = dict()
        if expansions is not None:
            for exp in expansions:
                if exp not in self.EXPANSIONS:
                    raise ValueError("Unknown expansion: {}".format(exp))
            params["expansions"] = ",".join(expansions)
        if tweet_fields is not None:
            for f in tweet_fields:
                if f not in self.TWEET_FIELDS:
                    raise ValueError("Unknown tweet field: {}".format(f))
            params["tweet.fields"] = ",".join(tweet_fields)
        if user_fields is not None:
            for f in user_fields:
                if f not in self.USER_FIELDS:
                    raise ValueError("Unknown user field: {}".format(f))
            params["user.fields"] = ",".join(user_fields)
        if media_fields is not None:
            for f in media_fields:
                if f not in self.MEDIA_FIELDS:
                    raise ValueError("Unknown media field: {}".format(f))
            params["media.fields"] = ",".join(media_fields)
        if place_fields is not None:
            for f in place_fields:
                if f not in self.PLACE_FIELDS:
                    raise ValueError("Unknown place field: {}".format(f))
            params["place.fields"] = ",".join(place_fields)
        if poll_fields is not None:
            for f in poll_fields:
                if f not in self.POLL_FIELDS:
                    raise ValueError("Unknown poll field: {}".format(f))
            params["poll.fields"] = ",".join(poll_fields)

        page = 0
        while True:
            if datetime.datetime.now() - self.cycle_start_time["tweet_lookup"] > datetime.timedelta(minutes=15):
                self.request_count["tweet_lookup"] = 0
                self.cycle_start_time["tweet_lookup"] = datetime.datetime.now()

            if self.request_count["tweet_lookup"] < self.RATE_LIMITS["tweet_lookup"][1]:
                try:
                    params["ids"] = ",".join(tweet_ids[page*100:min((page+1)*100, len(tweet_ids))])
                    json_response = self._connect_to_endpoint("tweet_lookup", params)
                except RuntimeError as err:
                    if err.args[1] == 429:  # Rate limit exceeded
                        logging.warning("Reach \"tweet lookup\" rate limit")
                        seconds_to_next_cycle = self.cycle_start_time["tweet_lookup"] + datetime.timedelta(minutes=15) - datetime.datetime.now()
                        time.sleep(max(10., seconds_to_next_cycle.total_seconds()))
                        continue
                    elif err.args[1] == 503:  # Service Unavailable
                        logging.warning("Service Unavailable")
                        time.sleep(60)
                        continue
                    else:
                        raise err

                json.dump(json_response, open(str(output_path / "retrieved_tweet_{:06d}.json".format(page)), "w"))
                page += 1
                self.request_count["tweet_lookup"] += 1
                time.sleep(self.RATE_LIMITS["tweet_lookup"][0])
            else:
                time.sleep(10)

            if page * 100 > len(tweet_ids):
                break

    def get_place_information(self, place_id: str) -> Dict:
        """ Get all information about a known place

        Args:
            place_id (str): ID of the place from twitter API

        Returns:
            place_info (Dict): detailed information about the known place
        """
        while True:
            if datetime.datetime.now() - self.cycle_start_time["place_information"] > datetime.timedelta(minutes=15):
                self.request_count["place_information"] = 0
                self.cycle_start_time["place_information"] = datetime.datetime.now()

            if self.request_count["place_information"] < self.RATE_LIMITS["place_information"][1]:
                try:
                    json_response = self._connect_to_endpoint_v1(
                        "place_information", self.ENDPOINT_URLS["place_information"].format(place_id=place_id)
                    )
                except RuntimeError as err:
                    if err.args[1] == 429:  # Rate limit exceeded
                        logging.warning("Reach \"place_information\" rate limit")
                        seconds_to_next_cycle = self.cycle_start_time["place_information"] + datetime.timedelta(minutes=15) - datetime.datetime.now()
                        time.sleep(max(10., seconds_to_next_cycle.total_seconds()))
                        continue
                    elif err.args[1] == 503:  # Service Unavailable
                        logging.warning("Service Unavailable")
                        time.sleep(60)
                        continue
                    elif err.args[1] == 404:  # Invalid Request
                        logging.warning("Invalid Request: {}".format(err.args[2]))
                        return None
                    else:
                        raise err

                self.request_count["place_information"] += 1
                return json_response
            else:
                time.sleep(10)

    @property
    def _headers(self):
        return {"Authorization": "Bearer {}".format(self.bearer_token)}

    def _connect_to_endpoint(self, endpoint: str, params: Dict[str, str]):
        retry_count = 0
        while True:
            try:
                response = requests.request("GET", self.ENDPOINT_URLS[endpoint], headers=self._headers, params=params)
                logging.info("Status of {} GET request: {}".format(endpoint, response.status_code))
                if response.status_code != 200:
                    raise RuntimeError(endpoint, response.status_code, response.text)
                return response.json()
            except requests.exceptions.ConnectionError:
                retry_count += 1
                logging.warning("Retry Connection (Count: {}): {}".format(retry_count, endpoint))
                time.sleep(10)
                continue

    def _connect_to_endpoint_v1(self, endpoint: str, url: str):
        retry_count = 0
        while True:
            try:
                response = requests.request("GET", url, headers=self._headers)
                logging.info("Status of {} GET request: {}".format(endpoint, response.status_code))
                if response.status_code != 200:
                    raise RuntimeError(endpoint, response.status_code, response.text)
                return response.json()
            except requests.exceptions.ConnectionError:
                retry_count += 1
                logging.warning("Retry Connection (Count: {}): {}".format(retry_count, endpoint))
                time.sleep(10)
                continue
