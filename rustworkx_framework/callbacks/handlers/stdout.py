"""Stdout callback handler for pretty console output."""

from typing import Any
from uuid import UUID

from ..base import BaseCallbackHandler

__all__ = ["StdoutCallbackHandler"]


class StdoutCallbackHandler(BaseCallbackHandler):
    """Prints events to stdout with colors and emojis.

    Similar to LangChain's verbose mode.
    """

    def __init__(
        self,
        color: bool = True,
        show_prompts: bool = False,
        show_outputs: bool = True,
        truncate_length: int = 200,
    ):
        self.color = color
        self.show_prompts = show_prompts
        self.show_outputs = show_outputs
        self.truncate_length = truncate_length
        self._indent = 0

    def _truncate(self, text: str) -> str:
        if len(text) <= self.truncate_length:
            return text
        return text[: self.truncate_length] + "..."

    def _print(self, message: str) -> None:
        indent = "  " * self._indent
        print(f"{indent}{message}")

    # === Run lifecycle ===

    def on_run_start(
        self,
        *,
        run_id: UUID,
        query: str,
        num_agents: int = 0,
        execution_order: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._print(f"🚀 Run started: {num_agents} agents")
        if execution_order:
            self._print(f"   Order: {' → '.join(execution_order)}")
        self._indent += 1

    def on_run_end(
        self,
        *,
        run_id: UUID,
        output: str,
        success: bool = True,
        error: BaseException | None = None,
        total_tokens: int = 0,
        total_time_ms: float = 0.0,
        executed_agents: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._indent = max(0, self._indent - 1)
        if success:
            self._print(f"✅ Run completed: {total_tokens} tokens, {total_time_ms:.0f}ms")
        else:
            self._print(f"❌ Run failed: {error}")

    # === Agent lifecycle ===

    def on_agent_start(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        agent_name: str = "",
        step_index: int = 0,
        prompt: str = "",
        **kwargs: Any,
    ) -> None:
        name = agent_name or agent_id
        self._print(f"▶️  [{step_index}] {name} started")
        if self.show_prompts and prompt:
            self._print(f"   Prompt: {self._truncate(prompt)}")
        self._indent += 1

    def on_agent_end(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        output: str,
        agent_name: str = "",
        step_index: int = 0,
        tokens_used: int = 0,
        duration_ms: float = 0.0,
        is_final: bool = False,
        **kwargs: Any,
    ) -> None:
        self._indent = max(0, self._indent - 1)
        name = agent_name or agent_id
        final_marker = " [FINAL]" if is_final else ""
        self._print(
            f"✅ [{step_index}] {name} completed: {tokens_used} tokens, "
            f"{duration_ms:.0f}ms{final_marker}"
        )
        if self.show_outputs and output:
            self._print(f"   Output: {self._truncate(output)}")

    def on_agent_error(
        self,
        error: BaseException,
        *,
        run_id: UUID,
        agent_id: str,
        error_type: str = "",
        will_retry: bool = False,
        attempt: int = 0,
        max_attempts: int = 0,
        **kwargs: Any,
    ) -> None:
        retry_info = f" (retry {attempt}/{max_attempts})" if will_retry else ""
        self._print(f"❌ {agent_id} error: {error}{retry_info}")

    # === Retry ===

    def on_retry(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        attempt: int,
        max_attempts: int = 0,
        delay_ms: float = 0.0,
        error: str = "",
        **kwargs: Any,
    ) -> None:
        self._print(f"🔄 {agent_id} retry {attempt}/{max_attempts} (delay: {delay_ms:.0f}ms)")

    # === Token streaming ===

    def on_llm_new_token(
        self,
        token: str,
        *,
        run_id: UUID,
        agent_id: str,
        is_first: bool = False,
        is_last: bool = False,
        **kwargs: Any,
    ) -> None:
        if is_first:
            print(f"  💬 {agent_id}: ", end="", flush=True)
        print(token, end="", flush=True)
        if is_last:
            print()  # newline

    # === Planning ===

    def on_plan_created(
        self,
        *,
        run_id: UUID,
        num_steps: int,
        execution_order: list[str],
        **kwargs: Any,
    ) -> None:
        self._print(f"📋 Plan: {num_steps} steps")

    def on_replan(
        self,
        *,
        run_id: UUID,
        reason: str,
        old_remaining: list[str],
        new_remaining: list[str],
        replan_count: int = 0,
        **kwargs: Any,
    ) -> None:
        self._print(f"🔄 Replan #{replan_count}: {reason}")

    # === Pruning/Fallback ===

    def on_prune(
        self,
        *,
        run_id: UUID,
        agent_id: str,
        reason: str,
        **kwargs: Any,
    ) -> None:
        self._print(f"✂️  Pruned {agent_id}: {reason}")

    def on_fallback(
        self,
        *,
        run_id: UUID,
        failed_agent_id: str,
        fallback_agent_id: str,
        reason: str = "",
        **kwargs: Any,
    ) -> None:
        self._print(f"🔀 Fallback: {failed_agent_id} → {fallback_agent_id}")

    # === Parallel execution ===

    def on_parallel_start(
        self,
        *,
        run_id: UUID,
        agent_ids: list[str],
        group_index: int = 0,
        **kwargs: Any,
    ) -> None:
        agents = ", ".join(agent_ids)
        self._print(f"⚡ Parallel group {group_index}: [{agents}]")
        self._indent += 1

    def on_parallel_end(
        self,
        *,
        run_id: UUID,
        agent_ids: list[str],
        group_index: int = 0,
        successful: list[str] | None = None,
        failed: list[str] | None = None,
        **kwargs: Any,
    ) -> None:
        self._indent = max(0, self._indent - 1)
        success_count = len(successful or [])
        total = len(agent_ids)
        self._print(f"⚡ Parallel group {group_index} done: {success_count}/{total}")

    # === Budget ===

    def on_budget_warning(
        self,
        *,
        run_id: UUID,
        budget_type: str,
        current: float,
        limit: float,
        ratio: float = 0.0,
        **kwargs: Any,
    ) -> None:
        self._print(f"⚠️  Budget warning: {budget_type} at {ratio:.0%}")

    def on_budget_exceeded(
        self,
        *,
        run_id: UUID,
        budget_type: str,
        current: float,
        limit: float,
        action_taken: str = "",
        **kwargs: Any,
    ) -> None:
        self._print(f"🛑 Budget exceeded: {budget_type} ({current:.0f}/{limit:.0f})")
