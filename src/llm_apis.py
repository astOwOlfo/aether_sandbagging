import asyncio
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any

import openai
from dotenv import load_dotenv
from openai import AsyncOpenAI
from openai.types.chat import ChatCompletion

load_dotenv()

CACHE_DIR = Path(".cache")

_INITIAL_RETRY_DELAY_SECONDS = 1.0
_MAX_RETRY_DELAY_SECONDS = 8 * 60


@dataclass(frozen=True, slots=True)
class Model:
    model: str
    thinking: bool = True
    max_tokens: int = 32768
    temperature: float = 1.0
    max_parallel: int = 256


class StopReason(Enum):
    LENGTH = "length"
    CONTENT_FILTER = "content_filter"
    TOOL_CALLS = "tool_calls"


@dataclass(frozen=True, slots=True)
class Completion:
    completion: str
    reasoning: str | None


class _RetryableResponseError(Exception):
    """The API returned a well-formed response that reports a provider failure."""


_client: AsyncOpenAI | None = None


def _get_client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
        )
    return _client


# Semaphores are keyed by (event loop, model name): the cap is per model name, and a
# semaphore must not be reused across event loops (e.g. successive asyncio.run calls).
_semaphores: dict[tuple[int, str], asyncio.Semaphore] = {}


def _get_semaphore(model: Model) -> asyncio.Semaphore:
    key = (id(asyncio.get_running_loop()), model.model)
    if key not in _semaphores:
        _semaphores[key] = asyncio.Semaphore(model.max_parallel)
    return _semaphores[key]


_total_input_tokens = 0
_total_output_tokens = 0


def _count_tokens(response: ChatCompletion) -> None:
    global _total_input_tokens, _total_output_tokens
    if response.usage is None:
        return
    previous_input, previous_output = _total_input_tokens, _total_output_tokens
    _total_input_tokens += response.usage.prompt_tokens
    _total_output_tokens += response.usage.completion_tokens
    if (
        previous_input // 1_000_000 < _total_input_tokens // 1_000_000
        or previous_output // 1_000_000 < _total_output_tokens // 1_000_000
    ):
        print(
            f"[llm_apis] tokens used since program start: "
            f"input={_total_input_tokens:,} output={_total_output_tokens:,}"
        )


def _cache_path(model: Model, messages: list[dict], seed: int) -> Path:
    key = {
        field.name: getattr(model, field.name)
        for field in fields(Model)
        if field.name != "max_parallel"
    }
    key["messages"] = messages
    key["seed"] = seed
    digest = hashlib.sha256(
        json.dumps(key, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    # Shard into subdirectories so no single directory grows huge; one file per key
    # means lookups stay O(1) regardless of cache size.
    return CACHE_DIR / digest[:2] / f"{digest}.json"


def _read_cache(path: Path) -> str | None:
    try:
        return path.read_text()
    except FileNotFoundError:
        return None


def _write_cache(path: Path, data: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Write to a unique temp file, then atomically rename, so concurrent readers and
    # writers (including other processes) never see a partially written file.
    tmp = path.with_name(f".tmp-{os.getpid()}-{uuid.uuid4().hex}")
    tmp.write_text(data)
    os.replace(tmp, path)


def _parse_response(response: ChatCompletion) -> Completion | StopReason:
    error = getattr(response, "error", None)
    if error is not None:
        raise _RetryableResponseError(f"error in response body: {error!r}")
    if not response.choices:
        raise ValueError(f"response has no choices: {response!r}")
    if len(response.choices) > 1:
        raise ValueError(f"response has multiple choices: {response!r}")
    choice = response.choices[0]
    choice_error = getattr(choice, "error", None)
    if choice_error is not None or choice.finish_reason == "error":
        raise _RetryableResponseError(f"error in response choice: {choice!r}")
    if choice.finish_reason is None:
        print("[debug] response with finish reason None:", response)
        raise _RetryableResponseError(
            f"finish reason is None in response choice: {choice}"
        )
    if choice.finish_reason != "stop":
        try:
            return StopReason(choice.finish_reason)
        except ValueError:
            raise ValueError(
                f"unexpected finish_reason {choice.finish_reason!r} in {choice!r}"
            ) from None
    content = choice.message.content
    if not isinstance(content, str) or content == "":
        print("[debug] response with empty content:", response)
        # raise ValueError(f"finish_reason is 'stop' but content is {content!r}")
        raise _RetryableResponseError(
            f"finish_reason is 'stop' but content is {content!r}"
        )
    reasoning = getattr(choice.message, "reasoning", None)
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError(f"reasoning is not a string: {reasoning!r}")
    if reasoning is not None and reasoning.strip() == "":
        reasoning = None
    return Completion(completion=content, reasoning=reasoning)


async def _call_api_with_retries(model: Model, messages: list[Any]) -> ChatCompletion:
    delay = _INITIAL_RETRY_DELAY_SECONDS
    while True:
        try:
            response = await _get_client().chat.completions.create(
                model=model.model,
                messages=messages,
                max_tokens=model.max_tokens,
                temperature=model.temperature,
                extra_body={
                    "reasoning": {"enabled": model.thinking},
                    "provider": {"sort": "price"},
                },
            )
            _count_tokens(response)
            # Raises _RetryableResponseError on provider failures reported inside the
            # response body; assumption violations (ValueError) propagate immediately.
            _parse_response(response)
            return response
        except (openai.APIError, _RetryableResponseError) as e:
            print(
                f"[llm_apis] API call to {model.model} failed "
                f"({type(e).__name__}: {e}), retrying in {delay:.0f}s"
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, _MAX_RETRY_DELAY_SECONDS)


async def generate(
    model: Model, prompt: str | list[dict], seed: int
) -> Completion | StopReason:
    """Generate a response via OpenRouter, with indefinite retries and disk caching.

    `seed` is only part of the cache key (to allow resampling despite caching); it is
    not passed to the API.
    """
    messages = (
        [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    )
    path = _cache_path(model, messages, seed)
    cached = await asyncio.to_thread(_read_cache, path)
    if cached is not None:
        return _parse_response(ChatCompletion.model_validate_json(cached))
    async with _get_semaphore(model):
        response = await _call_api_with_retries(model, messages)
    result = _parse_response(response)
    await asyncio.to_thread(_write_cache, path, response.model_dump_json())
    return result
