"""Transport-independent failure details for read-only integrations."""


class ReadFailure(Exception):
    def __init__(self, status_code: int):
        self.status_code = status_code
        self.code = {
            401: "authentication_required",
            403: "forbidden",
            404: "not_found",
            429: "rate_limited",
        }.get(status_code, "upstream_error")
        self.retryable = status_code in {408, 429} or status_code >= 500
        super().__init__(f"{self.code} (HTTP {status_code})")
