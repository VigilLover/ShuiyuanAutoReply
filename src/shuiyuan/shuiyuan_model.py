import io
import os
import re
import time
import base64
import pickle
import aiohttp
import asyncio
import hashlib
import logging
import traceback
import http.cookies
from datetime import datetime, timezone, timedelta
from PIL import Image
from dacite import from_dict
from typing import Optional, ClassVar, List, Dict, Any
from .constants import *
from .objects import *


class CookiesFileNotFoundError(Exception):
    """Custom exception for when the cookies file is not found."""

    pass


class CSRFTokenNotFoundError(Exception):
    """Custom exception for when the CSRF token is not found in the response."""

    pass


class ShuiyuanModel:
    """
    This class is used to interact with the Shuiyuan API.
    We should login to Shuiyuan here.
    """

    _shared_session: ClassVar[Optional[aiohttp.ClientSession]] = None
    _session_init_lock: ClassVar[Optional[asyncio.Lock]] = None
    _request_chain: ClassVar[Optional[asyncio.Future]] = None
    _request_interval: ClassVar[float] = 1.0
    _last_request_ts: ClassVar[float] = 0.0
    _active_instances: ClassVar[int] = 0

    def __init__(self):
        """
        Initialize the ShuiyuanModel.
        Use create() class method for async initialization.
        """
        self.session = None

    @classmethod
    async def create(cls, file_path: str = "cookies"):
        """
        Create and initialize a ShuiyuanModel instance with async operations.

        :param file_path: The path to the cookies file.
        :return: Initialized ShuiyuanModel instance.
        """
        # Ensure locks are initialized
        cls._ensure_locks()
        # Load the persistence cookie from the specified file path
        instance = cls()
        await instance._load_persistence_cookie(file_path)
        # Update the active instance count
        cls._active_instances += 1
        return instance

    @classmethod
    def _ensure_locks(cls) -> None:
        # Initialize locks if they are not already initialized
        if cls._session_init_lock is None:
            cls._session_init_lock = asyncio.Lock()
        if cls._request_chain is None:
            loop = asyncio.get_running_loop()
            future = loop.create_future()
            future.set_result(None)
            cls._request_chain = future

    @classmethod
    async def _ensure_shared_session(cls, file_path: str) -> aiohttp.ClientSession:
        # Lock to ensure only one session is created
        async with cls._session_init_lock:
            # If already initialized, return it
            if cls._shared_session is not None and not cls._shared_session.closed:
                return cls._shared_session

            # Check if the cookies file exists
            if not os.path.exists(file_path):
                raise CookiesFileNotFoundError(
                    "[FILESYSTEM] "
                    "Failed to find persistent cookies, "
                    "please run the ipynb file to get them first"
                )

            # Create a new aiohttp session and load cookies
            session = aiohttp.ClientSession()
            with open(file_path, "rb") as f:
                cookies = pickle.load(f)
                session.cookie_jar.update_cookies(cookies)

            # Update the shared session using Shuiyuan API
            cls._shared_session = session
            await cls._update_cookies()
            return cls._shared_session

    @classmethod
    async def _update_cookies(cls) -> None:
        if cls._shared_session is None:
            raise RuntimeError("Shared session is not initialized")

        # get the cookies
        cls._shared_session.headers.update({"User-Agent": default_user_agent})

        # try to get CSRF Token from response, with retries against network lag
        max_retries = 5
        format = r'<meta name="csrf-token" content="([^"]+)"[^>]*>'
        match = None

        for attempt in range(max_retries):
            response = await cls._rate_limited_request("get", get_cookies_url)
            text = await response.text()
            match = re.search(format, text)
            
            if match:
                break
                
            logging.warning(f"CSRF token not found on attempt {attempt + 1}. Retrying in 2 seconds...")
            await asyncio.sleep(2)

        if not match:
            raise CSRFTokenNotFoundError(
                "[INITIALIZATION] "
                "Failed to find CSRF token in the response after multiple attempts, "
                "please check the cookies file or the website structure"
            )

        # OK, let's update the CSRF token in the session headers
        csrf_token = match.group(1)
        cls._shared_session.headers.update({"X-CSRF-Token": csrf_token})

    async def _load_persistence_cookie(self, file_path: str) -> None:
        # load the shared session once and reuse it across instances
        self.session = await self._ensure_shared_session(file_path)

    @classmethod
    async def _rate_limited_request(cls, method: str, *args, **kwargs):
        # Check if the shared session is initialized
        if cls._shared_session is None:
            raise RuntimeError("Shared session is not initialized")

        # Get the previous future in the request chain
        # And create a new one for this request
        loop = asyncio.get_running_loop()
        wait_for = cls._request_chain
        if wait_for is None:
            wait_for = loop.create_future()
            wait_for.set_result(None)
        next_future = loop.create_future()
        cls._request_chain = next_future

        # Wait until the previous request is done
        await wait_for

        try:
            # Calculate the wait time to enforce rate limiting
            now = time.monotonic()
            wait_time = cls._request_interval - (now - cls._last_request_ts)
            if wait_time > 0:
                await asyncio.sleep(wait_time)

            # Make the actual request
            request_coro = getattr(cls._shared_session, method)
            response = await request_coro(*args, **kwargs)

            # Update the last request timestamp
            cls._last_request_ts = time.monotonic()
            return response
        finally:
            # Whatever happens, we need to set the next future result
            if not next_future.done():
                next_future.set_result(None)

    async def reply_to_post(
        self,
        raw: str,
        topic_id: int,
        reply_to_post_number: Optional[int] = None,
    ) -> None:
        """
        Reply to a topic with the given raw content.

        :param raw: The content to reply with.
        :param topic_id: The ID of the topic to reply to.
        :param reply_to_post_number: The post number to reply to.
        """

        # First we construct the form data we need to post
        form_data = aiohttp.FormData()
        form_data.add_field("raw", raw)
        form_data.add_field("topic_id", str(topic_id))
        if reply_to_post_number is not None:
            form_data.add_field("reply_to_post_number", str(reply_to_post_number))

        # OK, let's post it
        while True:
            response = await self._rate_limited_request(
                "post", reply_url, data=form_data
            )
            if response.status == 200:
                break
            elif response.status == 429:
                logging.warning(f"Failed to reply to post: {await response.text()}")
                await asyncio.sleep(1)
            else:
                raise Exception(f"Failed to reply to post: {await response.text()}")

    async def get_topic_details(self, topic_id: int) -> TopicDetails:
        """
        Get the details of a topic by its ID.

        :param topic_id: The ID of the topic to retrieve.
        :return: An instance of TopicDetails containing the topic information.
        """
        response = await self._rate_limited_request(
            "get", f"{get_topic_url}/{topic_id}.json"
        )
        if response.status != 200:
            raise Exception(f"Failed to get topic details: {await response.text()}")

        data = await response.json()
        return from_dict(TopicDetails, data)

    async def get_user_by_username(self, username: str) -> Optional[User]:
        """
        Get user details by username.

        :param username: The username of the user to retrieve.
        :return: An instance of User containing the user information.
        """
        response = await self._rate_limited_request(
            "get", f"{get_user_url}/{username}.json"
        )
        if response.status == 404:
            logging.warning(f"User '{username}' not found.")
            return None
        elif response.status != 200:
            raise Exception(f"Failed to get user details: {await response.text()}")

        data = await response.json()
        user_fields = data.get("user")
        if not user_fields:
            return None
        return from_dict(User, user_fields)

    async def get_post_details(self, post_id: int) -> PostDetails:
        """
        Get the details of a post by its ID.

        :param post_id: The ID of the post to retrieve.
        :return: An instance of PostDetails containing the post information.
        """
        response = await self._rate_limited_request(
            "get", f"{reply_url}/{post_id}.json"
        )
        if response.status != 200:
            raise Exception(f"Failed to get post details: {await response.text()}")

        data = await response.json()
        return from_dict(PostDetails, data)

    async def get_post_details_batch_by_topic_id(
        self, topic_id: int, post_ids: List[int]
    ) -> List[PostDetails]:
        """
        Get the details of all posts in a topic by its ID.

        :param topic_id: The ID of the topic to retrieve posts from.
        :param post_ids: A list of post IDs to retrieve details for.
        :return: A list of PostDetails instances containing the post information.
        """
        response = await self._rate_limited_request(
            "get",
            f"{get_topic_url}/{topic_id}/posts.json",
            params={"post_ids[]": post_ids},
        )
        data = await response.json()
        post_stream = data.get("post_stream", {})
        posts = post_stream.get("posts", [])
        return [from_dict(PostDetails, post_data) for post_data in posts]

    async def get_actions(self, username: str, filter: List[int], offset: int = 0) -> UserActions:
        """
        Get the latest actions for a given username and filter.

        :param username: The username to check actions for.
        :param filter: The list of action types to filter.
        :param offset: The offset of the actions, default to 0.
        :return: An instance of UserActions containing the mention information.
        """
        response = await self._rate_limited_request(
            "get",
            f"{action_url}",
            params={
                "offset": offset,
                "username": username,
                "filter": ",".join(map(str, filter)),
            },
        )
        if response.status != 200:
            raise Exception(f"Failed to get at notifications: {await response.text()}")

        data = await response.json()
        return from_dict(UserActions, data)

    async def upload_image(self, image_bytes: bytes) -> ImageUploadResponse:
        """
        Upload an image to the Shuiyuan server.

        :param image_bytes: The bytes of the image to upload.
        :return: The URL of the uploaded image.
        """
        form_data = aiohttp.FormData()
        form_data.add_field("upload_type", "composer")
        form_data.add_field("relative_path", "null")
        form_data.add_field("type", "image/jpeg")
        # Calculate the SHA1 checksum of the image
        sha1sum = hashlib.sha1(image_bytes).hexdigest()
        form_data.add_field("sha1sum", sha1sum)
        form_data.add_field(
            "file",
            image_bytes,
            filename="image.jpg",
            content_type="image/jpeg",
        )

        response = await self._rate_limited_request(
            "post", upload_url, data=form_data, timeout=10
        )
        if response.status != 200:
            raise Exception(f"Failed to upload image: {await response.text()}")

        data = await response.json()
        return from_dict(ImageUploadResponse, data)

    async def try_upload_image(
        self,
        image_bytes: bytes,
        try_base64: bool = True,
        try_base64_size_kb: int = 40,
    ) -> ImageURL:
        """
        Try to upload an image and return its URL or base64 HTML code.

        :param image_bytes: The bytes of the image to upload.
        :param try_base64: Whether to try converting to base64 if upload fails.
        :param try_base64_size_kb: The target size in KB for base64 conversion.
        :return: An ImageURL instance containing the URL or base64 HTML code.
        """
        try:
            # Upload the image and get the response
            response = await self.upload_image(image_bytes)
            return ImageURL("url", f"![img]({response.short_url})")
        except Exception as e:
            # If try_base64 is False, we will not try to convert later
            if not try_base64:
                logging.error(
                    f"Failed to upload image to Shuiyuan server, "
                    f"traceback is as follows:\n{traceback.format_exc()}"
                )
                raise e

            # Log the error and traceback
            logging.warning(
                f"Failed to upload image to Shuiyuan server, "
                f"traceback is as follows:\n{traceback.format_exc()}"
            )
            logging.warning("Trying to convert the image to base64 HTML code.")

            try:
                with Image.open(io.BytesIO(image_bytes)) as pil_image:
                    base64_image = self.compress_image_to_base64(
                        img=pil_image,
                        target_size_kb=try_base64_size_kb,
                    )
                    return ImageURL(
                        "base64",
                        f'<img alt="img" src="data:image/jpeg;base64,{base64_image}" />',
                    )
            except Exception as e:
                logging.error(
                    f"Failed to convert image to base64 HTML code, "
                    f"traceback is as follows:\n{traceback.format_exc()}"
                )
                raise e

    @staticmethod
    def compress_image_to_base64(
        img: Image.Image,
        target_size_kb: int = 20,
        quality: int = 100,
        step: int = 5,
    ):
        # Ensure image is in RGB mode
        if img.mode != "RGB":
            img = img.convert("RGB")

        buffer = io.BytesIO()
        current_quality = quality

        while current_quality > 0:
            buffer.seek(0)
            buffer.truncate()

            # Try to save the image with the current quality
            img.save(buffer, format="JPEG", quality=current_quality)
            size_kb = len(buffer.getvalue()) / 1024  # Size in KB

            # Check if the size is within the target
            if size_kb <= target_size_kb:
                break

            # Try to continue reducing quality
            current_quality -= step

        if current_quality <= 0:
            raise ValueError("Cannot compress image to the target size")

        # Convert to base64
        return base64.b64encode(buffer.getvalue()).decode("utf-8")

    async def close(self) -> None:
        """
        Close the aiohttp session and clean up resources.
        """
        await type(self)._decrement_instance_and_maybe_close()

    async def __aenter__(self):
        """
        Async context manager entry.
        """
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """
        Async context manager exit. Automatically closes the session.
        """
        await self.close()

    @classmethod
    async def _decrement_instance_and_maybe_close(cls) -> None:
        if cls._active_instances > 0:
            cls._active_instances -= 1

        if cls._active_instances == 0:
            await cls._close_shared_session()

    @classmethod
    async def _close_shared_session(cls) -> None:
        # Lock to ensure no re-creation during closing
        async with cls._session_init_lock:
            if cls._shared_session and not cls._shared_session.closed:
                await cls._shared_session.close()

        # Set shared session variables to None
        cls._shared_session = None
        cls._request_chain = None
        cls._last_request_ts = 0.0

    #################################################
    ##             Query Methods Start             ##
    #################################################

    async def search_user_by_term(self, term: str) -> List[User]:
        """
        Search for users by a search term.

        :param term: The search term to use for finding users. It has to be NON-EMPTY.
        :return: A list of User instances matching the search term.
        """
        response = await self._rate_limited_request(
            "get", f"{user_search_url}", params={"term": term, "limit": 6}
        )
        if response.status != 200:
            raise Exception(f"Failed to search users: {await response.text()}")

        data = await response.json()
        user_list = data.get("users", [])
        return [from_dict(User, user) for user in user_list]

    async def search_post_by_optional_username_topic(
        self, term: str, username: Optional[str] = None, topic_id: Optional[int] = None
    ) -> List[PostSearchResult]:
        """
        Search for posts by a search term and an optional username.

        :param term: The search term to use for finding posts. It has to be NON-EMPTY.
        :param username: An optional username to filter posts by.
        :param topic_id: An optional topic ID to filter posts by.
        :return: A list of PostSearchResult instances matching the search criteria.
        """
        # Construct the params
        params = {"term": term}
        if username:
            params["term"] += f" @{username}"
        if topic_id:
            params["search_context[type]"] = "topic"
            params["search_context[id]"] = str(topic_id)

        response = await self._rate_limited_request(
            "get", f"{post_search_url}", params=params
        )
        if response.status != 200:
            raise Exception(f"Failed to search posts: {await response.text()}")

        data = await response.json()
        post_list = data.get("posts", [])
        return [from_dict(PostSearchResult, post) for post in post_list]

    async def search_posts_by_time_range_and_topic(
        self, topic_id: int, after_date: Optional[str] = None, before_date: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for posts within a specific topic and time range, returning cleaned dictionary results.
        Note for AI Agents: The search API is exclusive of the dates provided. 
        If you want to search for posts exactly ON a single day (e.g., today's posts on '2026-03-18'), 
        you MUST set 'after_date' to the start day ('2026-03-18') AND set 'before_date' to the NEXT day ('2026-03-19').
        Failure to space them by at least one day (e.g. after_date='2026-03-18' and before_date='2026-03-18') will result in empty results.

        :param topic_id: The ID of the topic to search in.
        :param after_date: An optional start date (format: YYYY-MM-DD).
        :param before_date: An optional end date (format: YYYY-MM-DD). Must be at least 1 day after after_date to capture a single day.
        :return: A list of dicts with cleaned post content.
        """
        term = f"topic:{topic_id}"
        if after_date:
            term += f" after:{after_date}"
        if before_date:
            term += f" before:{before_date}"

        params = {"term": term.strip()}
        response = await self._rate_limited_request(
            "get", f"{post_search_url}", params=params
        )
        if response.status != 200:
            raise Exception(f"Failed to search posts by time range: {await response.text()}")

        data = await response.json()
        post_list = data.get("posts", [])
        posts = [from_dict(PostSearchResult, post) for post in post_list]
        return self._format_posts_to_dicts(posts)

    async def search_post_details_by_optional_username_topic(
        self, term: str, username: Optional[str] = None, topic_id: Optional[int] = None
    ) -> List[Dict[str, Any]]:
        """
        Search for posts by a search term and an optional username, returning cleaned dictionary results.

        :param term: The search term to use for finding posts. It has to be NON-EMPTY.
        :param username: An optional username to filter posts by.
        :param topic_id: An optional topic ID to filter posts by.
        :return: A list of dicts with cleaned post content.
        """
        post_search_results = await self.search_post_by_optional_username_topic(
            term, username, topic_id
        )
        post_details_list = []
        for result in post_search_results:
            try:
                details = await self.get_post_details(result.id)
                post_details_list.append(details)
            except Exception as e:
                logging.error(
                    f"Failed to get details for post ID {result.id}, "
                    f"traceback is as follows:\n{traceback.format_exc()}"
                )
        return self._format_posts_to_dicts(post_details_list)

    @staticmethod
    def process_replies(content: str) -> str:
        if not content:
            return ""
            
        aPattern = re.compile(r'<a.*?>.*?</a>')
        emojiPattern = re.compile(r'<img[^>]*title="(:[^:]*:)"[^>]*>')
        htmlPattern = re.compile(r'<[^>\n]*>')
        signaturePattern = re.compile(r'<div\s+data-signature[^>]*>.*?</div>', re.DOTALL | re.IGNORECASE)
        
        content = signaturePattern.sub('', content)
        content = re.sub(r'\[right\].*?\[/right\]', '', content, flags=re.IGNORECASE)
        content = content.replace('这里是中杯小狼(>^ω^<)', '')
        content = content.replace('这里是中杯小狼(&gt;^ω^&lt;)', '')
        content = content.replace('▶ \n总结\n', '').replace('▶\n总结\n', '').replace('▶ \n总结', '').replace('▶\n总结', '')
        content = content.replace('&hellip;', '')
        content = aPattern.sub('', content)
        
        while emojiPattern.search(content):
            content = emojiPattern.sub(lambda m: m.group(1), content, 1)
            
        content = htmlPattern.sub('', content)
        return content.strip()

    def _format_posts_to_dicts(self, posts: list) -> List[Dict[str, Any]]:
        filtered_res = []
        for post in posts:
            content = getattr(post, 'raw', None) or getattr(post, 'cooked', None) or getattr(post, 'blurb', "")
            clean_content = self.process_replies(content)
            
            # Convert UTC time string to UTC+8 formatted string
            created_at_utc8 = post.created_at
            try:
                # Discourse returns ISO 8601 UTC like "2026-02-15T04:22:06.264Z"
                dt_utc = datetime.strptime(post.created_at, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=timezone.utc)
                dt_utc8 = dt_utc.astimezone(timezone(timedelta(hours=8)))
                created_at_utc8 = dt_utc8.strftime("%Y-%m-%d %H:%M:%S")
            except ValueError:
                pass # fallback to original string if format is somehow different
            
            post_dict = {
                "username": post.username,
                "name": getattr(post, 'name', None),
                "created_at": created_at_utc8,
                "content": clean_content
            }
            filtered_res.append(post_dict)
        return filtered_res

    async def query_recent_posts_by_topic_id(
        self, topic_id: int, limit: int
    ) -> List[PostDetails]:
        """
        Query recent posts in a topic by its ID.

        :param topic_id: The ID of the topic to query.
        :param limit: The maximum number of recent posts to retrieve.
        :return: A list of PostDetails instances for the recent posts in the topic.
        """
        # Check if `limit` is positive
        if limit <= 0:
            raise ValueError("Limit must be a positive integer")

        # Retrieve the topic details to get the post stream
        topic_details = await self.get_topic_details(topic_id)

        # Use the last `limit` posts for recent activity
        recent_posts = topic_details.post_stream.stream[-limit:]
        return await self.get_post_details_batch_by_topic_id(topic_id, recent_posts)


def _global_ignore_illegal_cookies() -> None:
    # ignore the illegal key error
    http.cookies._is_legal_key = lambda _: True


_global_ignore_illegal_cookies()
