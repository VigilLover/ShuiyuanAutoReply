"""Safe URL quoting and bounded metadata for image download failures."""

from urllib.parse import quote

from yarl import URL


class ImageDownloadError(Exception):
    def __init__(self, status: int, retry_after: str | None = None):
        self.status = status
        self.retry_after = retry_after
        self.retryable = status in {408, 429} or status >= 500
        super().__init__(f"Image download HTTP {status}")


def encoded_image_url(url: str) -> URL:
    # Preserve existing escapes and signed query bytes; encode only unsafe/unicode chars.
    return URL(quote(url, safe=":/?#[]@!$&'()*+,;=%"), encoded=True)
