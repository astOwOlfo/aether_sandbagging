import asyncio
import hashlib
import json
import os
import uuid
from dataclasses import dataclass, fields
from enum import Enum
from pathlib import Path
from typing import Any, Literal

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
    max_parallel: int = 64
    provider: Literal["openrouter", "vllm"] = "openrouter"
    vllm_base_url: str = "http://localhost:8000/v1/"
    web_search: bool = False

    def __post_init__(self) -> None:
        if self.web_search and self.provider != "openrouter":
            raise ValueError(
                f"web_search is only supported with the openrouter provider, "
                f"not {self.provider!r}"
            )


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


# One client per base URL, so openrouter and vllm (possibly several servers) coexist.
_clients: dict[str, AsyncOpenAI] = {}


def _get_client(model: Model) -> AsyncOpenAI:
    if model.provider == "openrouter":
        base_url = "https://openrouter.ai/api/v1"
        api_key = os.environ["OPENROUTER_API_KEY"]
    else:
        base_url = model.vllm_base_url
        # vLLM only checks the key if started with --api-key; the OpenAI client
        # requires some non-empty string, and "EMPTY" is the vLLM convention.
        api_key = os.environ.get("VLLM_API_KEY", "EMPTY")
    if base_url not in _clients:
        _clients[base_url] = AsyncOpenAI(base_url=base_url, api_key=api_key)
    return _clients[base_url]


# Semaphores are keyed by (event loop, provider, model name): the cap is per model
# name within a provider, and a semaphore must not be reused across event loops
# (e.g. successive asyncio.run calls).
_semaphores: dict[tuple[int, str, str], asyncio.Semaphore] = {}


def _get_semaphore(model: Model) -> asyncio.Semaphore:
    key = (id(asyncio.get_running_loop()), model.provider, model.model)
    if key not in _semaphores:
        _semaphores[key] = asyncio.Semaphore(model.max_parallel)
    return _semaphores[key]


# In-flight requests keyed by (event loop, cache path): concurrent calls with the
# same cache key share one API call. Without this they would race on the cache file
# (last write wins), so on a rerun the losing call would return a value it never saw.
# Keyed by loop id for the same reason as _semaphores. The future holds the response
# JSON exactly as written to the cache.
_inflight: dict[tuple[int, Path], "asyncio.Future[str]"] = {}


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
        if field.name not in ("max_parallel", "vllm_base_url")
    }
    key["messages"] = messages
    key["seed"] = seed
    # Omit the default provider and web_search from the key so that cache entries
    # written before these fields existed stay valid. The vllm base url is
    # deliberately never part of the key: the same model served from a different
    # address is the same model.
    if key["provider"] == "openrouter":
        del key["provider"]
    if not key["web_search"]:
        del key["web_search"]
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
    if reasoning is None:
        # vLLM (with a --reasoning-parser) reports reasoning as `reasoning_content`.
        reasoning = getattr(choice.message, "reasoning_content", None)
    if reasoning is not None and not isinstance(reasoning, str):
        raise ValueError(f"reasoning is not a string: {reasoning!r}")
    if reasoning is not None and reasoning.strip() == "":
        reasoning = None
    return Completion(completion=content, reasoning=reasoning)


async def _call_api_with_retries(model: Model, messages: list[Any]) -> ChatCompletion:
    delay = _INITIAL_RETRY_DELAY_SECONDS
    # The `reasoning` and `provider` fields are OpenRouter extensions; vLLM controls
    # reasoning server-side (--reasoning-parser) and would reject unknown fields.
    extra_body: dict[str, Any] | None = None
    if model.provider == "openrouter":
        extra_body = {
            "reasoning": {"enabled": model.thinking},
            "provider": {"sort": "price"},
        }
        if model.web_search:
            extra_body["plugins"] = [{"id": "web"}]
    while True:
        try:
            response = await _get_client(model).chat.completions.create(
                model=model.model,
                messages=messages,
                max_tokens=model.max_tokens,
                temperature=model.temperature,
                extra_body=extra_body,
            )
            _count_tokens(response)
            # Raises _RetryableResponseError on provider failures reported inside the
            # response body; assumption violations (ValueError) propagate immediately.
            _parse_response(response)
            return response
        except (
            openai.APIError,
            _RetryableResponseError,
            json.decoder.JSONDecodeError,
        ) as e:
            print(
                f"[llm_apis] API call to {model.model} failed "
                f"({type(e).__name__}: {e}), retrying in {delay:.0f}s"
            )
            await asyncio.sleep(delay)
            delay = min(delay * 2, _MAX_RETRY_DELAY_SECONDS)


async def generate(
    model: Model,
    prompt: str | list[dict],
    seed: int,
) -> Completion | StopReason:
    """Generate a response, with indefinite retries and disk caching.

    Calls go to OpenRouter or to an OpenAI-compatible vLLM server at
    `model.vllm_base_url`, depending on `model.provider`. With vLLM, `model.thinking`
    has no effect: reasoning is configured server-side, and any reasoning the server
    parses out is returned in `Completion.reasoning`. `model.vllm_base_url` is not
    part of the cache key.

    `seed` is only part of the cache key (to allow resampling despite caching); it is
    not passed to the API.
    """
    messages = (
        [{"role": "user", "content": prompt}] if isinstance(prompt, str) else prompt
    )
    path = _cache_path(model, messages, seed)
    key = (id(asyncio.get_running_loop()), path)
    inflight = _inflight.get(key)
    if inflight is None:
        cached = await asyncio.to_thread(_read_cache, path)
        if cached is not None:
            return _parse_response(ChatCompletion.model_validate_json(cached))
        # The cache read yielded to the event loop, so an identical call may have
        # registered itself in the meantime; nothing yields between this re-check
        # and our own registration below.
        inflight = _inflight.get(key)
    if inflight is not None:
        # shield: cancelling one waiter must not cancel the shared future.
        return _parse_response(
            ChatCompletion.model_validate_json(await asyncio.shield(inflight))
        )
    future: asyncio.Future[str] = asyncio.get_running_loop().create_future()
    _inflight[key] = future
    try:
        async with _get_semaphore(model):
            response = await _call_api_with_retries(model, messages)
        result = _parse_response(response)
        data = response.model_dump_json()
        await asyncio.to_thread(_write_cache, path, data)
    except BaseException as e:
        future.set_exception(e)
        future.exception()  # mark retrieved, so no-waiter failures don't log a warning
        raise
    finally:
        # On success this runs after the cache write, so late duplicates that miss
        # the registry are guaranteed to hit the cache instead of calling the API
        # again; on failure it lets them retry from scratch.
        del _inflight[key]
    future.set_result(data)
    return result
