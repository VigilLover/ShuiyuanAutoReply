"""Raw-text OpenAI-compatible embeddings without tokenizer or Torch dependencies."""
import asyncio
import math
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

from langchain_core.embeddings import Embeddings
from openai import APIConnectionError, APIStatusError, AsyncOpenAI, OpenAI


class OpenAIEmbeddings(Embeddings):
    def __init__(self, settings, *, client=None, async_client=None):
        self.settings = settings
        kwargs = dict(api_key=settings['api_key'], base_url=settings['base_url'],
                      timeout=settings['timeout'], max_retries=0)
        self.client = client or OpenAI(**kwargs)
        self.async_client = async_client or AsyncOpenAI(**kwargs)
        self.semaphore = asyncio.Semaphore(settings['concurrency'])

    def _request(self, texts):
        if any(not isinstance(t, str) or not t.strip() for t in texts):
            raise ValueError('Embedding input must contain nonempty text')
        return dict(model=self.settings['model'], input=texts,
                    dimensions=self.settings['dims'], encoding_format='float')

    def _validate(self, response, count):
        data = sorted(response.data, key=lambda item: item.index)
        if [item.index for item in data] != list(range(count)):
            raise ValueError('Embedding response indices do not match input')
        result = [item.embedding for item in data]
        if any(len(v) != self.settings['dims'] or any(not math.isfinite(x) for x in v) for v in result):
            raise ValueError('Embedding response contains invalid dimensions or values')
        return result

    @staticmethod
    def _delay(exc, attempt):
        retryable = isinstance(exc, APIConnectionError) or (
            isinstance(exc, APIStatusError) and (exc.status_code == 429 or exc.status_code >= 500))
        if not retryable:
            raise exc
        raw = getattr(getattr(exc, 'response', None), 'headers', {}).get('retry-after')
        if raw:
            try:
                return max(0, float(raw))
            except ValueError:
                try:
                    return max(0, (parsedate_to_datetime(raw) - datetime.now(timezone.utc)).total_seconds())
                except (TypeError, ValueError):
                    pass
        return min(2 ** attempt, 8)

    def embed_documents(self, texts):
        result = []
        batch = self.settings['batch_size']
        for offset in range(0, len(texts), batch):
            items = texts[offset:offset + batch]
            for attempt in range(self.settings['attempts']):
                try:
                    response = self.client.embeddings.create(**self._request(items))
                    result.extend(self._validate(response, len(items)))
                    break
                except (APIConnectionError, APIStatusError) as exc:
                    delay = self._delay(exc, attempt)
                    if attempt + 1 == self.settings['attempts']:
                        raise
                    time.sleep(delay)
        return result

    async def aembed_documents(self, texts):
        result = []
        batch = self.settings['batch_size']
        for offset in range(0, len(texts), batch):
            items = texts[offset:offset + batch]
            for attempt in range(self.settings['attempts']):
                try:
                    async with self.semaphore:
                        response = await self.async_client.embeddings.create(**self._request(items))
                    result.extend(self._validate(response, len(items)))
                    break
                except (APIConnectionError, APIStatusError) as exc:
                    delay = self._delay(exc, attempt)
                    if attempt + 1 == self.settings['attempts']:
                        raise
                    await asyncio.sleep(delay)
        return result

    def embed_query(self, text):
        return self.embed_documents([text])[0]

    async def aembed_query(self, text):
        return (await self.aembed_documents([text]))[0]

    async def aclose(self):
        self.client.close()
        await self.async_client.close()
