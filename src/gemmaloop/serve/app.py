"""OpenAI-compatible HTTP server for Gemma 4 / Gemma-Loop.

``POST /v1/chat/completions`` accepts the OpenAI Chat Completions payload
(text, OpenAI image parts as data URLs, native tool declarations), streams SSE
or returns JSON, and forwards reasoning in ``message.reasoning_content`` and
tool calls in ``message.tool_calls``.  ``GET /v1/models`` advertises the local
model so OpenCode and lm-eval-style clients can discover it.

Requests are micro-batched: a background worker drains the queue for at most
``wait_ms`` and runs up to ``max_batch`` requests that share the same sampling
configuration, tools, seed and image token budget through one
:func:`gemmaloop.serve.generate.generate_batch` call.  The model is loaded
lazily on the first request so importing the app (or running the tests with an
injected ``gen_fn``/``loader``) never touches the GPU.

CLI::

    uv run python -m gemmaloop.serve.app --size e2b --quant bf16 \\
        --loops 2 --bridge scalar --adapter <dir> [--adapter <dir2>] --port 8000
"""

from __future__ import annotations

import argparse
import asyncio
import contextlib
import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from gemmaloop.model.layer_plan import parse_block
from gemmaloop.model.load import ModelSpec, _resolve_loop_config, load_model
from gemmaloop.model.looped import LoopConfig
from gemmaloop.serve.generate import (
    GenOut,
    Sampling,
    generate_batch,
    normalize_messages,
)

MODEL_ID = "gemma-loop"
DEFAULT_MAX_NEW_TOKENS = 16384

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Micro-batcher
# ---------------------------------------------------------------------------


@dataclass
class _Job:
    conversation: list[dict]
    sampling: Sampling
    tools: list[dict] | None
    seed: int
    image_soft_tokens: int
    future: asyncio.Future
    key: tuple = field(compare=False)


@dataclass
class _State:
    spec: ModelSpec
    max_batch: int
    wait_ms: int
    gen_fn: Callable
    loader: Callable
    model: Any = None
    processor: Any = None
    plan: Any = None
    loaded: bool = False
    load_lock: asyncio.Lock = field(default_factory=asyncio.Lock)
    queue: asyncio.Queue | None = None
    worker: asyncio.Task | None = None


def _job_key(
    sampling: Sampling,
    tools: list[dict] | None,
    seed: int,
    image_soft_tokens: int,
) -> tuple:
    return (
        sampling.temperature,
        sampling.top_p,
        sampling.top_k,
        sampling.max_new_tokens,
        sampling.enable_thinking,
        json.dumps(tools, sort_keys=True) if tools else "",
        seed,
        image_soft_tokens,
    )


async def _ensure_loaded(state: _State) -> None:
    if state.loaded:
        return
    async with state.load_lock:
        if state.loaded:
            return
        state.model, state.processor, state.plan = await asyncio.to_thread(
            state.loader, state.spec
        )
        state.loaded = True


def _ensure_worker(state: _State) -> None:
    if state.worker is not None and not state.worker.done():
        return
    state.queue = asyncio.Queue()
    state.worker = asyncio.get_running_loop().create_task(_worker(state))


async def _worker(state: _State) -> None:
    while True:
        first = await state.queue.get()
        batch = [first]
        deadline = asyncio.get_running_loop().time() + state.wait_ms / 1000.0
        while len(batch) < state.max_batch:
            remaining = deadline - asyncio.get_running_loop().time()
            if remaining <= 0:
                break
            try:
                batch.append(
                    await asyncio.wait_for(state.queue.get(), timeout=remaining)
                )
            except asyncio.TimeoutError:
                break

        groups: dict[tuple, list[_Job]] = {}
        for job in batch:
            groups.setdefault(job.key, []).append(job)
        for group in groups.values():
            await _run_group(state, group)


async def _run_group(state: _State, group: list[_Job]) -> None:
    try:
        await _ensure_loaded(state)
        head = group[0]
        outputs = await asyncio.to_thread(
            state.gen_fn,
            state.model,
            state.processor,
            [job.conversation for job in group],
            head.sampling,
            tools=head.tools,
            seed=head.seed,
            image_soft_tokens=head.image_soft_tokens,
        )
        if len(outputs) != len(group):
            raise RuntimeError(
                f"generator returned {len(outputs)} outputs for {len(group)} requests"
            )
    except Exception as exc:  # noqa: BLE001 - forwarded to every waiting request
        for job in group:
            if not job.future.done():
                job.future.set_exception(exc)
        return

    for job, output in zip(group, outputs):
        if not job.future.done():
            job.future.set_result(output)


async def _submit(
    state: _State,
    conversation: list[dict],
    sampling: Sampling,
    tools: list[dict] | None,
    seed: int,
    image_soft_tokens: int,
) -> GenOut:
    _ensure_worker(state)
    future = asyncio.get_running_loop().create_future()
    job = _Job(
        conversation=conversation,
        sampling=sampling,
        tools=tools,
        seed=seed,
        image_soft_tokens=image_soft_tokens,
        future=future,
        key=_job_key(sampling, tools, seed, image_soft_tokens),
    )
    await state.queue.put(job)
    return await future


# ---------------------------------------------------------------------------
# OpenAI payload conversion
# ---------------------------------------------------------------------------


def _sampling_from_payload(payload: dict) -> Sampling:
    chat_template_kwargs = payload.get("chat_template_kwargs") or {}
    enable_thinking = payload.get("enable_thinking")
    if enable_thinking is None:
        enable_thinking = chat_template_kwargs.get("enable_thinking", True)
    max_new_tokens = payload.get("max_completion_tokens")
    if max_new_tokens is None:
        max_new_tokens = payload.get("max_tokens")
    if max_new_tokens is None:
        max_new_tokens = DEFAULT_MAX_NEW_TOKENS
    return Sampling(
        temperature=float(payload.get("temperature", 1.0)),
        top_p=float(payload.get("top_p", 0.95)),
        top_k=int(payload.get("top_k", 64)),
        max_new_tokens=int(max_new_tokens),
        enable_thinking=bool(enable_thinking),
    )


def _tools_from_payload(payload: dict) -> list[dict] | None:
    if payload.get("tool_choice") == "none":
        return None
    tools = payload.get("tools")
    if not tools:
        return None
    if not isinstance(tools, list):
        raise ValueError("tools must be a list")
    return tools


def _completion_message(output: GenOut, include_reasoning: bool = True) -> dict:
    message: dict[str, Any] = {"role": "assistant", "content": output.text}
    if include_reasoning:
        message["reasoning_content"] = output.reasoning
    if output.tool_calls:
        message["tool_calls"] = output.tool_calls
    if output.invalid_tool_call:
        message["x_invalid_tool_call"] = True
    return message


def _usage(output: GenOut) -> dict:
    return {
        "prompt_tokens": output.prompt_tokens,
        "completion_tokens": output.completion_tokens,
        "total_tokens": output.prompt_tokens + output.completion_tokens,
    }


def _completion_body(output: GenOut, model_name: str) -> dict:
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    return {
        "id": completion_id,
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": _completion_message(output),
                "finish_reason": output.finish_reason,
            }
        ],
        "usage": _usage(output),
    }


def _sse(payload: dict | str) -> str:
    if isinstance(payload, str):
        return f"data: {payload}\n\n"
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


def _stream_events(output: GenOut, model_name: str, include_usage: bool = False):
    completion_id = f"chatcmpl-{uuid.uuid4().hex}"
    created = int(time.time())

    def chunk(delta: dict, finish_reason: str | None = None) -> dict:
        return {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model_name,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }

    yield _sse(chunk({"role": "assistant", "content": ""}))
    if output.reasoning:
        yield _sse(chunk({"reasoning_content": output.reasoning}))
    if output.text:
        yield _sse(chunk({"content": output.text}))
    if output.tool_calls:
        yield _sse(
            chunk(
                {
                    "tool_calls": [
                        {"index": index, **call}
                        for index, call in enumerate(output.tool_calls)
                    ]
                }
            )
        )
    yield _sse(chunk({}, output.finish_reason))
    if include_usage:
        # OpenAI's terminal usage event: no choices, just the counters.
        yield _sse(
            {
                "id": completion_id,
                "object": "chat.completion.chunk",
                "created": created,
                "model": model_name,
                "choices": [],
                "usage": _usage(output),
            }
        )
    yield _sse("[DONE]")


# ---------------------------------------------------------------------------
# FastAPI application
# ---------------------------------------------------------------------------


def create_app(
    spec: ModelSpec,
    max_batch: int = 64,
    wait_ms: int = 50,
    gen_fn: Callable | None = None,
    loader: Callable = load_model,
) -> FastAPI:
    """Build the app.

    ``gen_fn`` and ``loader`` are injectable so tests (and Task 9) can drive the
    server without loading a checkpoint.  ``gen_fn`` must have the
    :func:`generate_batch` signature.  The loader runs once, lazily, on the
    first completion request.
    """
    if max_batch < 1:
        raise ValueError("max_batch must be at least 1")
    if wait_ms < 0:
        raise ValueError("wait_ms must be non-negative")

    state = _State(
        spec=spec,
        max_batch=max_batch,
        wait_ms=wait_ms,
        gen_fn=gen_fn or generate_batch,
        loader=loader,
    )

    @contextlib.asynccontextmanager
    async def lifespan(app: FastAPI):
        yield
        worker = state.worker
        if worker is not None and not worker.done():
            worker.cancel()
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await worker

    app = FastAPI(title="Gemma-Loop", version="0.1.0", lifespan=lifespan)

    @app.post("/v1/chat/completions")
    async def chat_completions(request: Request):
        try:
            payload = await request.json()
        except Exception as exc:  # noqa: BLE001 - malformed JSON body
            raise HTTPException(status_code=400, detail="invalid JSON body") from exc
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="body must be a JSON object")

        messages = payload.get("messages")
        if not isinstance(messages, list) or not messages:
            raise HTTPException(
                status_code=400, detail="messages must be a non-empty list"
            )

        try:
            sampling = _sampling_from_payload(payload)
            tools = _tools_from_payload(payload)
            conversation = normalize_messages(messages)
            seed = int(payload.get("seed") or 0)
            image_soft_tokens = int(payload.get("image_soft_tokens", 280))
        except (TypeError, ValueError, KeyError) as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc

        try:
            output = await _submit(
                state, conversation, sampling, tools, seed, image_soft_tokens
            )
        except Exception as exc:  # noqa: BLE001 - loader/generator failure
            raise HTTPException(status_code=500, detail=str(exc)) from exc

        if output.invalid_tool_call:
            logger.warning(
                "invalid tool call in completion; raw output kept as content"
            )

        if payload.get("stream"):
            stream_options = payload.get("stream_options") or {}
            include_usage = bool(stream_options.get("include_usage", False))
            return StreamingResponse(
                _stream_events(output, MODEL_ID, include_usage=include_usage),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache"},
            )
        return JSONResponse(_completion_body(output, MODEL_ID))

    @app.get("/v1/models")
    async def list_models():
        return {
            "object": "list",
            "data": [
                {
                    "id": MODEL_ID,
                    "object": "model",
                    "created": 0,
                    "owned_by": "gemma-loop",
                }
            ],
        }

    return app


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_spec(args: argparse.Namespace) -> ModelSpec:
    adapters = tuple(args.adapter or ())
    loop = None
    if args.loops is not None:
        bridge = args.bridge
        if bridge is None:
            # Omitted --bridge: inherit it from the adapter's loop state.
            adapter_loop = _resolve_loop_config(None, adapters)
            bridge = adapter_loop.bridge if adapter_loop is not None else "scalar"
        # Validate at build time (startup), not on the first request.  N is
        # fixed for the whole server process.
        block_text = getattr(args, "block", None)
        loop = _resolve_loop_config(
            LoopConfig(
                n_loops=args.loops,
                bridge=bridge,
                block=parse_block(block_text) if block_text else None,
                adapter_policy=getattr(args, "adapter_policy", None),
                lora_target_layers=getattr(args, "lora_target_layers", None),
            ),
            adapters,
        )
    return ModelSpec(
        size=args.size,
        quant=args.quant,
        adapters=adapters,
        loop=loop,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Serve Gemma-Loop through an OpenAI-compatible API."
    )
    parser.add_argument("--size", choices=("e2b", "e4b"), required=True)
    parser.add_argument("--quant", choices=("bf16", "nf4"), default="bf16")
    parser.add_argument("--adapter", action="append", default=[])
    parser.add_argument(
        "--loops",
        type=int,
        default=None,
        help=(
            "N for the looped block (omit to serve the unpatched model; an "
            "adapter carrying loop state is still patched at N=1)"
        ),
    )
    parser.add_argument(
        "--bridge",
        choices=("scalar", "none", "damped", "gch"),
        default=None,
        help="loop bridge (default: the adapter's bridge, else scalar)",
    )
    parser.add_argument(
        "--block",
        default=None,
        help=(
            "inclusive loop block START-END inside the KV-consumer region "
            "(default: the adapter's block, else legacy 15-29)"
        ),
    )
    parser.add_argument(
        "--adapter-policy",
        default=None,
        choices=("always", "loop_only"),
        help="LoRA activation policy (default: the adapter's, else always)",
    )
    parser.add_argument(
        "--lora-target-layers",
        default=None,
        choices=("language", "loop_block", "loop_block_coda"),
        help="LoRA target layer set (default: the adapter's, else language)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-batch", type=int, default=8)
    parser.add_argument("--wait-ms", type=int, default=50)
    args = parser.parse_args(argv)

    import uvicorn

    app = create_app(_build_spec(args), max_batch=args.max_batch, wait_ms=args.wait_ms)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
