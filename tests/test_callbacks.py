from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest

from rustworkx_framework.callbacks import (
    AsyncCallbackHandler,
    AsyncCallbackManager,
    BaseCallbackHandler,
    CallbackManager,
    get_callback_manager,
    set_callback_manager,
    trace_as_callback,
)


@dataclass
class Call:
    name: str
    kwargs: dict[str, Any] = field(default_factory=dict)


class RecordingHandler(BaseCallbackHandler):
    def __init__(self) -> None:
        self.calls: list[Call] = []

    def on_run_start(
        self,
        *,
        run_id: UUID,
        query: str,
        num_agents: int = 0,
        execution_order: list[str] | None = None,
        parent_run_id: UUID | None = None,
        tags: list[str] | None = None,
        metadata: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        self.calls.append(
            Call(
                "on_run_start",
                {
                    "run_id": run_id,
                    "query": query,
                    "num_agents": num_agents,
                    "execution_order": execution_order,
                    "parent_run_id": parent_run_id,
                    "tags": tags,
                    "metadata": metadata,
                    **kwargs,
                },
            )
        )

    def on_retry(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        attempt: int,
        max_attempts: int = 0,
        delay_ms: float = 0.0,
        error: str = "",
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.calls.append(
            Call(
                "on_retry",
                {
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "delay_ms": delay_ms,
                    "error": error,
                    "parent_run_id": parent_run_id,
                    **kwargs,
                },
            )
        )

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        agent_id: str,
        token_index: int = 0,
        is_first: bool = False,
        is_last: bool = False,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.calls.append(
            Call(
                "on_llm_new_token",
                {
                    "token": token,
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "token_index": token_index,
                    "is_first": is_first,
                    "is_last": is_last,
                    "parent_run_id": parent_run_id,
                    **kwargs,
                },
            )
        )

    def on_memory_read(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        entries_count: int = 0,
        keys: list[str] | None = None,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.calls.append(
            Call(
                "on_memory_read",
                {
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "entries_count": entries_count,
                    "keys": keys,
                    "parent_run_id": parent_run_id,
                    **kwargs,
                },
            )
        )

    def on_budget_warning(
        self,
        *,
        run_id: UUID,
        budget_type: str,
        current: float,
        limit: float,
        ratio: float = 0.0,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.calls.append(
            Call(
                "on_budget_warning",
                {
                    "run_id": run_id,
                    "budget_type": budget_type,
                    "current": current,
                    "limit": limit,
                    "ratio": ratio,
                    "parent_run_id": parent_run_id,
                    **kwargs,
                },
            )
        )


class RaisingHandler(BaseCallbackHandler):
    def __init__(self, *, raise_error: bool) -> None:
        self.raise_error = raise_error

    def on_run_start(self, **_: Any) -> None:
        msg = "boom"
        raise RuntimeError(msg)


class AsyncRecordingHandler(AsyncCallbackHandler):
    def __init__(self) -> None:
        self.calls: list[Call] = []

    async def on_retry(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        attempt: int,
        max_attempts: int = 0,
        delay_ms: float = 0.0,
        error: str = "",
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        self.calls.append(
            Call(
                "on_retry",
                {
                    "run_id": run_id,
                    "agent_id": agent_id,
                    "attempt": attempt,
                    "max_attempts": max_attempts,
                    "delay_ms": delay_ms,
                    "error": error,
                    "parent_run_id": parent_run_id,
                    **kwargs,
                },
            )
        )


def test_trace_as_callback_sets_and_resets_context_manager() -> None:
    set_callback_manager(None)
    assert get_callback_manager() is None

    with trace_as_callback(tags=["outer"], metadata={"x": 1}) as outer:
        assert get_callback_manager() is outer
        assert outer.tags == ["outer"]
        assert outer.metadata == {"x": 1}

        with trace_as_callback(tags=["inner"]) as inner:
            assert get_callback_manager() is inner
            assert inner.tags == ["inner"]

        assert get_callback_manager() is outer

    assert get_callback_manager() is None


def test_callback_manager_on_run_start_passes_parent_tags_metadata() -> None:
    h = RecordingHandler()
    parent = uuid4()
    m = CallbackManager.configure(handlers=[h], tags=["t1"], metadata={"k": "v"})
    m.parent_run_id = parent

    run_id = m.on_run_start(query="hi", num_agents=2)

    assert len(h.calls) == 1
    call = h.calls[0]
    assert call.name == "on_run_start"
    assert call.kwargs["run_id"] == run_id
    assert call.kwargs["query"] == "hi"
    assert call.kwargs["num_agents"] == 2
    assert call.kwargs["execution_order"] == []
    assert call.kwargs["parent_run_id"] == parent
    assert call.kwargs["tags"] == ["t1"]
    assert call.kwargs["metadata"] == {"k": "v"}


def test_callback_manager_ignore_flags_filter_events() -> None:
    run_id = uuid4()

    h_ok = RecordingHandler()
    h_ignore_retry = RecordingHandler()
    h_ignore_retry.ignore_retry = True

    h_ignore_llm = RecordingHandler()
    h_ignore_llm.ignore_llm = True

    h_ignore_memory = RecordingHandler()
    h_ignore_memory.ignore_memory = True

    h_ignore_budget = RecordingHandler()
    h_ignore_budget.ignore_budget = True

    m = CallbackManager(handlers=[h_ok, h_ignore_retry, h_ignore_llm, h_ignore_memory, h_ignore_budget])

    m.on_retry(run_id, agent_id="a1", attempt=1, max_attempts=3, delay_ms=10.0, error="e")
    m.on_llm_new_token(run_id, "tok", agent_id="a1", token_index=0, is_first=True, is_last=False)
    m.on_memory_read(run_id, agent_id="a1", entries_count=2, keys=["k1", "k2"])
    m.on_budget_warning(run_id, budget_type="tokens", current=90.0, limit=100.0, ratio=0.9)

    assert [c.name for c in h_ok.calls] == [
        "on_retry",
        "on_llm_new_token",
        "on_memory_read",
        "on_budget_warning",
    ]
    assert [c.name for c in h_ignore_retry.calls] == [
        "on_llm_new_token",
        "on_memory_read",
        "on_budget_warning",
    ]
    assert [c.name for c in h_ignore_llm.calls] == [
        "on_retry",
        "on_memory_read",
        "on_budget_warning",
    ]
    assert [c.name for c in h_ignore_memory.calls] == [
        "on_retry",
        "on_llm_new_token",
        "on_budget_warning",
    ]
    assert [c.name for c in h_ignore_budget.calls] == [
        "on_retry",
        "on_llm_new_token",
        "on_memory_read",
    ]


def test_callback_manager_error_in_one_handler_does_not_block_others_when_raise_error_false() -> None:
    bad = RaisingHandler(raise_error=False)
    good = RecordingHandler()
    m = CallbackManager(handlers=[bad, good])

    run_id = m.on_run_start(query="q")

    assert len(good.calls) == 1
    assert good.calls[0].name == "on_run_start"
    assert good.calls[0].kwargs["run_id"] == run_id


def test_callback_manager_error_propagates_when_raise_error_true() -> None:
    bad = RaisingHandler(raise_error=True)
    good = RecordingHandler()
    m = CallbackManager(handlers=[bad, good])

    with pytest.raises(RuntimeError, match="boom"):
        m.on_run_start(query="q")

    assert good.calls == []


def test_callback_manager_get_child_inherits_only_inheritable_settings() -> None:
    inheritable = RecordingHandler()
    non_inheritable = RecordingHandler()

    m = CallbackManager()
    m.add_handler(inheritable, inherit=True)
    m.add_handler(non_inheritable, inherit=False)
    m.add_tags(["tag_inherit"], inherit=True)
    m.add_tags(["tag_local"], inherit=False)
    m.add_metadata({"a": 1}, inherit=True)
    m.add_metadata({"b": 2}, inherit=False)

    parent_run_id = uuid4()
    child = m.get_child(parent_run_id)

    assert child.parent_run_id == parent_run_id
    assert child.handlers == [inheritable]
    assert child.tags == ["tag_inherit"]
    assert child.metadata == {"a": 1}

    run_id = child.on_run_start(query="x")
    assert inheritable.calls[-1].kwargs["parent_run_id"] == parent_run_id
    assert inheritable.calls[-1].kwargs["run_id"] == run_id


@pytest.mark.asyncio
async def test_async_callback_manager_runs_sync_and_async_handlers() -> None:
    sync_h = RecordingHandler()
    async_h = AsyncRecordingHandler()
    m = AsyncCallbackManager(handlers=[sync_h, async_h])

    run_id = uuid4()
    await m.on_retry(run_id, agent_id="a1", attempt=2, max_attempts=5, delay_ms=1.0, error="x")

    assert [c.name for c in sync_h.calls] == ["on_retry"]
    assert [c.name for c in async_h.calls] == ["on_retry"]
