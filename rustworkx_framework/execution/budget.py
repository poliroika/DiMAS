"""
Управление бюджетами токенов, запросов и времени.

Обеспечивает контроль расходов на уровне графа и отдельных узлов.
"""

from collections.abc import Callable
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict

__all__ = [
    "Budget",
    "BudgetConfig",
    "BudgetTracker",
    "NodeBudget",
]


class Budget(BaseModel):
    """Учёт лимита с учётом использованных и зарезервированных ресурсов."""

    limit: float
    used: float = 0.0
    reserved: float = 0.0

    @property
    def available(self) -> float:
        """Оставшийся доступный ресурс (без учёта зарезервированного)."""
        return max(0.0, self.limit - self.used - self.reserved)

    @property
    def remaining(self) -> float:
        """Оставшийся ресурс без учёта резерва (limit - used)."""
        return max(0.0, self.limit - self.used)

    @property
    def usage_ratio(self) -> float:
        """Доля использованного ресурса в пределах лимита."""
        if self.limit <= 0:
            return 0.0
        return self.used / self.limit

    @property
    def is_exhausted(self) -> bool:
        """True, если свободного ресурса нет."""
        return self.available <= 0

    def can_spend(self, amount: float) -> bool:
        """Проверить, хватает ли доступного ресурса на указанную трату."""
        return self.available >= amount

    def spend(self, amount: float) -> bool:
        """Потратить ресурс, если хватает; вернуть успех операции."""
        if not self.can_spend(amount):
            return False
        self.used += amount
        return True

    def reserve(self, amount: float) -> bool:
        """Зарезервировать ресурс под будущую операцию."""
        if self.available < amount:
            return False
        self.reserved += amount
        return True

    def release_reservation(self, amount: float) -> None:
        """Освободить часть резерва."""
        self.reserved = max(0.0, self.reserved - amount)

    def commit_reservation(self, amount: float) -> None:
        """Списать из резерва в использование (до указанной суммы)."""
        actual = min(amount, self.reserved)
        self.reserved -= actual
        self.used += actual

    def reset(self) -> None:
        """Сбросить использованное и зарезервированное."""
        self.used = 0.0
        self.reserved = 0.0

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать бюджет в словарь."""
        return {
            "limit": self.limit,
            "used": self.used,
            "reserved": self.reserved,
            "available": self.available,
            "usage_ratio": self.usage_ratio,
        }


class NodeBudget(BaseModel):
    """Лимиты для конкретного узла (токены, запросы, время, длины)."""

    node_id: str
    tokens: Budget | None = None
    requests: Budget | None = None
    time_seconds: Budget | None = None
    max_prompt_length: int | None = None
    max_response_length: int | None = None

    def can_execute(self, estimated_tokens: int = 0) -> tuple[bool, str | None]:
        """Проверить, можно ли выполнить шаг с оценкой токенов."""
        if self.tokens and not self.tokens.can_spend(estimated_tokens):
            return False, f"Token budget exhausted for node {self.node_id}"

        if self.requests and not self.requests.can_spend(1):
            return False, f"Request budget exhausted for node {self.node_id}"

        return True, None

    def record_usage(
        self,
        tokens: int = 0,
        time_seconds: float = 0.0,
    ) -> None:
        """Записать фактическое потребление ресурсов узла."""
        if self.tokens:
            self.tokens.spend(tokens)
        if self.requests:
            self.requests.spend(1)
        if self.time_seconds:
            self.time_seconds.spend(time_seconds)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать бюджет узла в словарь."""
        return {
            "node_id": self.node_id,
            "tokens": self.tokens.to_dict() if self.tokens else None,
            "requests": self.requests.to_dict() if self.requests else None,
            "time_seconds": self.time_seconds.to_dict() if self.time_seconds else None,
            "limits": {
                "max_prompt_length": self.max_prompt_length,
                "max_response_length": self.max_response_length,
            },
        }


class BudgetConfig(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)
    """Конфигурация глобальных и покомпонентных лимитов на выполнение."""

    total_token_limit: int | None = None
    total_request_limit: int | None = None
    total_time_limit_seconds: float | None = None

    node_token_limit: int | None = None
    node_request_limit: int | None = None
    node_time_limit_seconds: float | None = None

    max_prompt_length: int | None = None
    max_response_length: int | None = None

    warn_at_usage_ratio: float = 0.8

    on_budget_warning: Callable[[str, Budget], None] | None = None
    on_budget_exceeded: Callable[[str, Budget], None] | None = None


class BudgetTracker:
    """Трекер бюджетов: глобальных и помодульных, с предупреждениями."""

    def __init__(self, config: BudgetConfig | None = None):
        self.config = config or BudgetConfig()

        self._global_tokens = Budget(limit=float(self.config.total_token_limit or float("inf")))
        self._global_requests = Budget(limit=float(self.config.total_request_limit or float("inf")))
        self._global_time = Budget(limit=self.config.total_time_limit_seconds or float("inf"))

        self._node_budgets: dict[str, NodeBudget] = {}
        self._start_time: datetime | None = None

    def start(self) -> None:
        """Зафиксировать время старта для контроля time-бюджета."""
        self._start_time = datetime.now()

    def get_elapsed_seconds(self) -> float:
        """Сколько секунд прошло с момента start()."""
        if self._start_time is None:
            return 0.0
        return (datetime.now() - self._start_time).total_seconds()

    def get_node_budget(self, node_id: str) -> NodeBudget:
        """Вернуть (или создать) бюджет для заданного узла."""
        if node_id not in self._node_budgets:
            self._node_budgets[node_id] = NodeBudget(
                node_id=node_id,
                tokens=Budget(limit=float(self.config.node_token_limit or float("inf")))
                if self.config.node_token_limit
                else None,
                requests=Budget(limit=float(self.config.node_request_limit or float("inf")))
                if self.config.node_request_limit
                else None,
                time_seconds=Budget(limit=self.config.node_time_limit_seconds or float("inf"))
                if self.config.node_time_limit_seconds
                else None,
                max_prompt_length=self.config.max_prompt_length,
                max_response_length=self.config.max_response_length,
            )
        return self._node_budgets[node_id]

    def can_execute(
        self,
        node_id: str,
        estimated_tokens: int = 0,
    ) -> tuple[bool, str | None]:
        """Проверить возможность выполнения шага с учётом глобальных и узловых лимитов."""
        if self._global_time.is_exhausted:
            elapsed = self.get_elapsed_seconds()
            time_limit = self.config.total_time_limit_seconds
            if time_limit is not None and elapsed >= time_limit:
                return False, f"Time budget exhausted: {elapsed:.1f}s"

        if not self._global_tokens.can_spend(estimated_tokens):
            return (
                False,
                f"Global token budget exhausted: {self._global_tokens.used}/{self._global_tokens.limit}",
            )

        if not self._global_requests.can_spend(1):
            return (
                False,
                f"Global request budget exhausted: {self._global_requests.used}/{self._global_requests.limit}",
            )

        node_budget = self.get_node_budget(node_id)
        can, reason = node_budget.can_execute(estimated_tokens)
        if not can:
            return False, reason

        return True, None

    def record_usage(
        self,
        node_id: str,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_seconds: float = 0.0,
    ) -> None:
        """Записать фактическое потребление по узлу и глобально."""
        total_tokens = prompt_tokens + completion_tokens

        self._global_tokens.spend(total_tokens)
        self._global_requests.spend(1)

        node_budget = self.get_node_budget(node_id)
        node_budget.record_usage(tokens=total_tokens, time_seconds=latency_seconds)

        self._check_warnings()

    def truncate_prompt(self, prompt: str) -> str:
        """Обрезать prompt по лимиту, добавив пометку."""
        if self.config.max_prompt_length and len(prompt) > self.config.max_prompt_length:
            return prompt[: self.config.max_prompt_length] + "\n[TRUNCATED]"
        return prompt

    def truncate_response(self, response: str) -> str:
        """Обрезать response по лимиту, добавив пометку."""
        if self.config.max_response_length and len(response) > self.config.max_response_length:
            return response[: self.config.max_response_length] + "\n[TRUNCATED]"
        return response

    def _check_warnings(self) -> None:
        """Вызывать callbacks, если достигнут порог warn_at_usage_ratio."""
        if self.config.on_budget_warning:
            if self._global_tokens.usage_ratio >= self.config.warn_at_usage_ratio:
                self.config.on_budget_warning("tokens", self._global_tokens)
            if self._global_requests.usage_ratio >= self.config.warn_at_usage_ratio:
                self.config.on_budget_warning("requests", self._global_requests)

    @property
    def global_tokens(self) -> Budget:
        return self._global_tokens

    @property
    def global_requests(self) -> Budget:
        return self._global_requests

    @property
    def global_time(self) -> Budget:
        return self._global_time

    def get_summary(self) -> dict[str, Any]:
        """Итоговая сводка по глобальным и узловым бюджетам."""
        return {
            "global": {
                "tokens": self._global_tokens.to_dict(),
                "requests": self._global_requests.to_dict(),
                "time": self._global_time.to_dict(),
                "elapsed_seconds": self.get_elapsed_seconds(),
            },
            "nodes": {node_id: budget.to_dict() for node_id, budget in self._node_budgets.items()},
        }

    def reset(self) -> None:
        """Сбросить все бюджеты и стартовое время."""
        self._global_tokens.reset()
        self._global_requests.reset()
        self._global_time.reset()
        self._node_budgets.clear()
        self._start_time = None
