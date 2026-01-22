"""Система памяти агентов с уровнями, шарингом и скрытыми каналами.

Поддерживает:
- Стратификацию (working/long-term memory)
- TTL и ограничения размера
- Компрессию и усечение
- Шаринг между подграфами с фильтрами доступа
- Внешние хранилища
- Скрытые каналы (hidden_state, embeddings)
"""

import time
from abc import ABC, abstractmethod
from collections.abc import Callable
from enum import Enum
from typing import Any, Protocol, runtime_checkable

import torch
from pydantic import BaseModel, ConfigDict, Field

__all__ = [
    "MemoryEntry",
    "MemoryLevel",
    "MemoryConfig",
    "AgentMemory",
    "AccessFilter",
    "SharingPolicy",
    "SharedMemoryPool",
    "MemoryStorage",
    "AsyncMemoryStorage",
    "Message",
    "HiddenChannel",
    "MessageProtocol",
    "CompressionStrategy",
    "TruncateCompressor",
    "SummaryCompressor",
]


class MemoryLevel(str, Enum):
    WORKING = "working"
    LONG_TERM = "long_term"
    SHARED = "shared"


class MemoryEntry(BaseModel):
    """Элемент памяти с TTL, приоритетом и метаданными."""

    content: dict[str, Any]
    level: MemoryLevel = MemoryLevel.WORKING
    created_at: float = Field(default_factory=time.time)
    accessed_at: float = Field(default_factory=time.time)
    ttl: float | None = None
    priority: int = 0
    tags: set[str] = Field(default_factory=set)
    source_agent: str | None = None

    @property
    def is_expired(self) -> bool:
        """True, если ttl истёк."""
        if self.ttl is None:
            return False
        return time.time() - self.created_at > self.ttl

    def touch(self) -> None:
        """Обновить отметку последнего доступа."""
        self.accessed_at = time.time()

    @property
    def age(self) -> float:
        """Возраст записи в секундах."""
        return time.time() - self.created_at


class CompressionStrategy(ABC):
    """Базовая стратегия компрессии набора MemoryEntry."""

    @abstractmethod
    def compress(
        self,
        entries: list[MemoryEntry],
        max_entries: int,
    ) -> list[MemoryEntry]: ...


class TruncateCompressor(CompressionStrategy):
    """Компрессор: оставляет ограниченное число записей по приоритету/свежести."""

    def __init__(self, keep_recent: bool = True, keep_high_priority: bool = True):
        self.keep_recent = keep_recent
        self.keep_high_priority = keep_high_priority

    def compress(
        self,
        entries: list[MemoryEntry],
        max_entries: int,
    ) -> list[MemoryEntry]:
        """Отсечь записи сверх лимита согласно стратегии сортировки."""
        if len(entries) <= max_entries:
            return entries

        def sort_key(e: MemoryEntry) -> tuple[int, float]:
            priority = -e.priority if self.keep_high_priority else 0
            age = -e.accessed_at if self.keep_recent else e.accessed_at
            return (priority, age)

        sorted_entries = sorted(entries, key=sort_key)
        return sorted_entries[:max_entries]


class SummaryCompressor(CompressionStrategy):
    """Компрессор: сворачивает старые записи в summary через summarizer."""

    def __init__(
        self,
        summarizer: Callable[[list[dict[str, Any]]], dict[str, Any]] | None = None,
        batch_size: int = 5,
    ):
        self.summarizer = summarizer
        self.batch_size = batch_size

    def compress(
        self,
        entries: list[MemoryEntry],
        max_entries: int,
    ) -> list[MemoryEntry]:
        """Сжать записи, оставив итоговый summary + самые свежие."""
        if len(entries) <= max_entries or self.summarizer is None:
            return entries[-max_entries:] if len(entries) > max_entries else entries

        to_keep = entries[-(max_entries - 1) :]
        to_summarize = entries[: -(max_entries - 1)]

        summary_content = self.summarizer([e.content for e in to_summarize])

        summary_entry = MemoryEntry(
            content=summary_content,
            level=MemoryLevel.LONG_TERM,
            priority=max(e.priority for e in to_summarize) if to_summarize else 0,
            tags=set().union(*(e.tags for e in to_summarize)),
        )

        return [summary_entry] + to_keep


class MemoryConfig(BaseModel):
    """Настройки объёмов, TTL и стратегии компрессии памяти агента."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    working_max_entries: int = 20
    working_default_ttl: float | None = 3600.0
    long_term_max_entries: int = 100
    long_term_default_ttl: float | None = None
    compression_strategy: CompressionStrategy = Field(default_factory=lambda: TruncateCompressor())
    auto_compress: bool = True
    promote_after_accesses: int = 3
    demote_inactive_after: float = 7200.0
    cleanup_interval: float = 300.0
    hidden_state_dim: int | None = None


class AgentMemory:
    """Менеджер памяти агента с уровнями working/long-term и скрытыми каналами."""

    def __init__(
        self,
        agent_id: str,
        config: MemoryConfig | None = None,
    ):
        self.agent_id = agent_id
        self.config = config or MemoryConfig()

        self._working: list[MemoryEntry] = []
        self._long_term: list[MemoryEntry] = []
        self._access_counts: dict[int, int] = {}
        self._last_cleanup = time.time()

        self._hidden_state: torch.Tensor | None = None
        self._embedding: torch.Tensor | None = None

    @property
    def working_memory(self) -> list[MemoryEntry]:
        """Актуальные записи рабочей памяти (без просроченных)."""
        self._maybe_cleanup()
        return [e for e in self._working if not e.is_expired]

    @property
    def long_term_memory(self) -> list[MemoryEntry]:
        """Актуальные записи долговременной памяти (без просроченных)."""
        return [e for e in self._long_term if not e.is_expired]

    @property
    def all_entries(self) -> list[MemoryEntry]:
        """Все записи обеих зон (с фильтрацией ttl)."""
        return self.working_memory + self.long_term_memory

    @property
    def hidden_state(self) -> torch.Tensor | None:
        return self._hidden_state

    @hidden_state.setter
    def hidden_state(self, value: torch.Tensor | None) -> None:
        self._hidden_state = value

    @property
    def embedding(self) -> torch.Tensor | None:
        return self._embedding

    @embedding.setter
    def embedding(self, value: torch.Tensor | None) -> None:
        self._embedding = value

    def add(
        self,
        content: dict[str, Any],
        level: MemoryLevel = MemoryLevel.WORKING,
        ttl: float | None = None,
        priority: int = 0,
        tags: set[str] | None = None,
        source_agent: str | None = None,
    ) -> MemoryEntry:
        """Добавить запись в выбранный уровень памяти."""
        if ttl is None:
            if level == MemoryLevel.WORKING:
                ttl = self.config.working_default_ttl
            else:
                ttl = self.config.long_term_default_ttl

        entry = MemoryEntry(
            content=content,
            level=level,
            ttl=ttl,
            priority=priority,
            tags=tags or set(),
            source_agent=source_agent,
        )

        if level == MemoryLevel.WORKING:
            self._working.append(entry)
            self._maybe_compress_working()
        else:
            self._long_term.append(entry)
            self._maybe_compress_long_term()

        return entry

    def add_message(
        self,
        role: str,
        content: str,
        **metadata,
    ) -> MemoryEntry:
        """Удобный метод для добавления чат-сообщения."""
        return self.add(
            content={"role": role, "content": content, **metadata},
            level=MemoryLevel.WORKING,
        )

    def get(
        self,
        level: MemoryLevel | None = None,
        tags: set[str] | None = None,
        limit: int | None = None,
        include_expired: bool = False,
    ) -> list[MemoryEntry]:
        """Получить записи с фильтрами по уровню/тегам/лимиту."""
        if level == MemoryLevel.WORKING:
            entries = self._working
        elif level == MemoryLevel.LONG_TERM:
            entries = self._long_term
        else:
            entries = self._working + self._long_term

        if not include_expired:
            entries = [e for e in entries if not e.is_expired]

        if tags:
            entries = [e for e in entries if tags & e.tags]

        for entry in entries:
            entry.touch()
            entry_id = id(entry)
            self._access_counts[entry_id] = self._access_counts.get(entry_id, 0) + 1

            if (
                entry.level == MemoryLevel.WORKING
                and self._access_counts[entry_id] >= self.config.promote_after_accesses
            ):
                self._promote(entry)

        if limit:
            entries = entries[-limit:]

        return entries

    def get_messages(self, limit: int | None = None) -> list[dict[str, Any]]:
        """Получить контенты записей, содержащих поле role (сообщения)."""
        entries = self.get(limit=limit)
        return [e.content for e in entries if "role" in e.content]

    def clear(self, level: MemoryLevel | None = None) -> None:
        """Очистить указанный уровень или всё."""
        if level is None or level == MemoryLevel.WORKING:
            self._working.clear()
        if level is None or level == MemoryLevel.LONG_TERM:
            self._long_term.clear()
        self._access_counts.clear()

    def remove_expired(self) -> int:
        """Удалить просроченные записи, вернуть количество удалённых."""
        before = len(self._working) + len(self._long_term)
        self._working = [e for e in self._working if not e.is_expired]
        self._long_term = [e for e in self._long_term if not e.is_expired]
        after = len(self._working) + len(self._long_term)
        return before - after

    def _promote(self, entry: MemoryEntry) -> None:
        if entry in self._working:
            self._working.remove(entry)
            entry.level = MemoryLevel.LONG_TERM
            entry.ttl = self.config.long_term_default_ttl
            self._long_term.append(entry)

    def _demote(self, entry: MemoryEntry) -> None:
        if entry in self._long_term:
            self._long_term.remove(entry)
            entry.level = MemoryLevel.WORKING
            entry.ttl = self.config.working_default_ttl
            self._working.append(entry)

    def _maybe_compress_working(self) -> None:
        if not self.config.auto_compress:
            return
        if len(self._working) > self.config.working_max_entries:
            self._working = self.config.compression_strategy.compress(
                self._working, self.config.working_max_entries
            )

    def _maybe_compress_long_term(self) -> None:
        if not self.config.auto_compress:
            return
        if len(self._long_term) > self.config.long_term_max_entries:
            self._long_term = self.config.compression_strategy.compress(
                self._long_term, self.config.long_term_max_entries
            )

    def _maybe_cleanup(self) -> None:
        now = time.time()
        if now - self._last_cleanup < self.config.cleanup_interval:
            return

        self._last_cleanup = now
        self.remove_expired()

        inactive_threshold = now - self.config.demote_inactive_after
        for entry in list(self._long_term):
            if entry.accessed_at < inactive_threshold:
                self._demote(entry)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать память агента для сохранения."""
        return {
            "agent_id": self.agent_id,
            "working": [
                {
                    "content": e.content,
                    "created_at": e.created_at,
                    "ttl": e.ttl,
                    "priority": e.priority,
                    "tags": list(e.tags),
                }
                for e in self._working
            ],
            "long_term": [
                {
                    "content": e.content,
                    "created_at": e.created_at,
                    "ttl": e.ttl,
                    "priority": e.priority,
                    "tags": list(e.tags),
                }
                for e in self._long_term
            ],
            "hidden_state": (
                self._hidden_state.cpu().tolist() if self._hidden_state is not None else None
            ),
            "embedding": (self._embedding.cpu().tolist() if self._embedding is not None else None),
        }

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        config: MemoryConfig | None = None,
    ) -> "AgentMemory":
        """Восстановить AgentMemory из словаря."""
        memory = cls(data["agent_id"], config)

        for e_data in data.get("working", []):
            entry = MemoryEntry(
                content=e_data["content"],
                level=MemoryLevel.WORKING,
                created_at=e_data.get("created_at", time.time()),
                ttl=e_data.get("ttl"),
                priority=e_data.get("priority", 0),
                tags=set(e_data.get("tags", [])),
            )
            memory._working.append(entry)

        for e_data in data.get("long_term", []):
            entry = MemoryEntry(
                content=e_data["content"],
                level=MemoryLevel.LONG_TERM,
                created_at=e_data.get("created_at", time.time()),
                ttl=e_data.get("ttl"),
                priority=e_data.get("priority", 0),
                tags=set(e_data.get("tags", [])),
            )
            memory._long_term.append(entry)

        if data.get("hidden_state"):
            memory._hidden_state = torch.tensor(data["hidden_state"])
        if data.get("embedding"):
            memory._embedding = torch.tensor(data["embedding"])

        return memory


class AccessFilter(ABC):
    """Интерфейс фильтра доступа к шареной памяти."""

    @abstractmethod
    def can_access(
        self,
        requester_id: str,
        owner_id: str,
        entry: MemoryEntry,
    ) -> bool: ...


class SharingPolicy(str, Enum):
    NONE = "none"
    SAME_SUBGRAPH = "same_subgraph"
    BY_TAGS = "by_tags"
    BY_ROLE_FAMILY = "by_role_family"
    FULL = "full"


class TagBasedFilter(AccessFilter):
    """Фильтр, разрешающий доступ по пересечению тегов."""

    def __init__(self, shared_tags: set[str]):
        self.shared_tags = shared_tags

    def can_access(
        self,
        requester_id: str,
        owner_id: str,
        entry: MemoryEntry,
    ) -> bool:
        return bool(entry.tags & self.shared_tags)


class SubgraphFilter(AccessFilter):
    """Фильтр доступа по принадлежности к одному подграфу."""

    def __init__(self, subgraph_members: dict[str, set[str]]):
        self.subgraph_members = subgraph_members
        self._agent_to_subgraph: dict[str, str] = {}
        for sg_id, members in subgraph_members.items():
            for member in members:
                self._agent_to_subgraph[member] = sg_id

    def can_access(
        self,
        requester_id: str,
        owner_id: str,
        entry: MemoryEntry,
    ) -> bool:
        req_sg = self._agent_to_subgraph.get(requester_id)
        own_sg = self._agent_to_subgraph.get(owner_id)
        return req_sg is not None and req_sg == own_sg


class RoleFamilyFilter(AccessFilter):
    """Фильтр доступа по семействам ролей."""

    def __init__(self, role_families: dict[str, set[str]]):
        self.role_families = role_families
        self._agent_to_family: dict[str, str] = {}
        for family, members in role_families.items():
            for member in members:
                self._agent_to_family[member] = family

    def can_access(
        self,
        requester_id: str,
        owner_id: str,
        entry: MemoryEntry,
    ) -> bool:
        req_fam = self._agent_to_family.get(requester_id)
        own_fam = self._agent_to_family.get(owner_id)
        return req_fam is not None and req_fam == own_fam


class SharedMemoryPool:
    """Пул шареной памяти с политикой доступа/распространения."""

    def __init__(
        self,
        access_filter: AccessFilter | None = None,
        default_policy: SharingPolicy = SharingPolicy.BY_TAGS,
    ):
        self.access_filter = access_filter
        self.default_policy = default_policy
        self._memories: dict[str, AgentMemory] = {}
        self._shared_entries: list[MemoryEntry] = []

    def register(self, memory: AgentMemory) -> None:
        self._memories[memory.agent_id] = memory

    def unregister(self, agent_id: str) -> None:
        self._memories.pop(agent_id, None)

    def share(
        self,
        from_agent: str,
        entry: MemoryEntry,
        to_agents: list[str] | None = None,
    ) -> None:
        """Поделиться записью с конкретными агентами или положить в общий пул."""
        shared_entry = MemoryEntry(
            content=entry.content.copy(),
            level=MemoryLevel.SHARED,
            ttl=entry.ttl,
            priority=entry.priority,
            tags=entry.tags.copy(),
            source_agent=from_agent,
        )

        if to_agents:
            for agent_id in to_agents:
                if agent_id in self._memories:
                    self._memories[agent_id].add(
                        content=shared_entry.content,
                        level=MemoryLevel.WORKING,
                        priority=shared_entry.priority,
                        tags=shared_entry.tags | {"shared"},
                        source_agent=from_agent,
                    )
        else:
            self._shared_entries.append(shared_entry)

    def get_shared(
        self,
        requester_id: str,
        tags: set[str] | None = None,
        limit: int | None = None,
    ) -> list[MemoryEntry]:
        """Получить записи из общего пула с учётом фильтра доступа."""
        entries = []

        for entry in self._shared_entries:
            if self.access_filter and entry.source_agent:
                if not self.access_filter.can_access(requester_id, entry.source_agent, entry):
                    continue

            if tags and not (tags & entry.tags):
                continue

            if not entry.is_expired:
                entries.append(entry)

        if limit:
            entries = entries[-limit:]

        return entries

    def get_from_agent(
        self,
        requester_id: str,
        owner_id: str,
        level: MemoryLevel | None = None,
        tags: set[str] | None = None,
    ) -> list[MemoryEntry]:
        """Получить записи конкретного агента с проверкой фильтра доступа."""
        if owner_id not in self._memories:
            return []

        owner_memory = self._memories[owner_id]
        entries = owner_memory.get(level=level, tags=tags)

        if self.access_filter:
            entries = [
                e for e in entries if self.access_filter.can_access(requester_id, owner_id, e)
            ]

        return entries

    def broadcast(
        self,
        from_agent: str,
        content: dict[str, Any],
        tags: set[str] | None = None,
    ) -> None:
        """Рассылать запись всем зарегистрированным агентам (кроме отправителя)."""
        for agent_id, memory in self._memories.items():
            if agent_id != from_agent:
                memory.add(
                    content=content,
                    level=MemoryLevel.WORKING,
                    tags=(tags or set()) | {"broadcast"},
                    source_agent=from_agent,
                )


@runtime_checkable
class MemoryStorage(Protocol):
    def save_memory(self, agent_id: str, memory: AgentMemory) -> None: ...

    def load_memory(self, agent_id: str) -> AgentMemory | None: ...

    def delete_memory(self, agent_id: str) -> None: ...

    def list_agents(self) -> list[str]: ...


@runtime_checkable
class AsyncMemoryStorage(Protocol):
    async def save_memory(self, agent_id: str, memory: AgentMemory) -> None: ...

    async def load_memory(self, agent_id: str) -> AgentMemory | None: ...

    async def delete_memory(self, agent_id: str) -> None: ...

    async def list_agents(self) -> list[str]: ...


class HiddenChannel(BaseModel):
    """Контейнер скрытого состояния/эмбеддинга с метаданными."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    hidden_state: torch.Tensor | None = None
    embedding: torch.Tensor | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Сериализовать скрытый канал в dict."""
        return {
            "hidden_state": (
                self.hidden_state.cpu().tolist() if self.hidden_state is not None else None
            ),
            "embedding": (self.embedding.cpu().tolist() if self.embedding is not None else None),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HiddenChannel":
        """Восстановить скрытый канал из словаря."""
        return cls(
            hidden_state=(torch.tensor(data["hidden_state"]) if data.get("hidden_state") else None),
            embedding=(torch.tensor(data["embedding"]) if data.get("embedding") else None),
            metadata=data.get("metadata", {}),
        )


class Message(BaseModel):
    """Сообщение с видимой частью и опциональным скрытым каналом."""

    sender_id: str
    receiver_id: str | None
    content: str
    role: str = "assistant"
    timestamp: float = Field(default_factory=time.time)
    hidden: HiddenChannel | None = None
    message_type: str = "response"
    priority: int = 0
    tags: set[str] = Field(default_factory=set)

    @property
    def has_hidden(self) -> bool:
        """True, если есть скрытый state/embedding."""
        return self.hidden is not None and (
            self.hidden.hidden_state is not None or self.hidden.embedding is not None
        )

    def to_visible_dict(self) -> dict[str, Any]:
        """Только видимая часть сообщения (role/content/sender)."""
        return {
            "role": self.role,
            "content": self.content,
            "sender": self.sender_id,
        }

    def to_full_dict(self) -> dict[str, Any]:
        """Полное представление сообщения с метаданными и скрытым каналом."""
        result = self.to_visible_dict()
        result["timestamp"] = self.timestamp
        result["message_type"] = self.message_type
        result["priority"] = self.priority
        result["tags"] = list(self.tags)
        if self.hidden:
            result["hidden"] = self.hidden.to_dict()
        return result


class MessageProtocol:
    """Протокол сообщений: создаёт сообщения, комбинирует скрытые состояния."""

    def __init__(
        self,
        enable_hidden: bool = True,
        hidden_dim: int | None = None,
        combine_hidden: Callable[[list[torch.Tensor]], torch.Tensor] | None = None,
    ):
        self.enable_hidden = enable_hidden
        self.hidden_dim = hidden_dim
        self.combine_hidden = combine_hidden or self._default_combine

    @staticmethod
    def _default_combine(tensors: list[torch.Tensor]) -> torch.Tensor:
        """Среднее скрытых тензоров — реализация по умолчанию."""
        if not tensors:
            raise ValueError("No tensors to combine")
        stacked = torch.stack(tensors)
        return stacked.mean(dim=0)

    def create_message(
        self,
        sender_id: str,
        content: str,
        receiver_id: str | None = None,
        hidden_state: torch.Tensor | None = None,
        embedding: torch.Tensor | None = None,
        **kwargs,
    ) -> Message:
        """Создать Message, опционально упаковывая скрытые данные."""
        hidden = None
        if self.enable_hidden and (hidden_state is not None or embedding is not None):
            hidden = HiddenChannel(
                hidden_state=hidden_state,
                embedding=embedding,
            )

        return Message(
            sender_id=sender_id,
            receiver_id=receiver_id,
            content=content,
            hidden=hidden,
            **kwargs,
        )

    def extract_hidden_states(
        self,
        messages: list[Message],
    ) -> list[torch.Tensor]:
        """Извлечь список скрытых состояний из сообщений."""
        states = []
        for msg in messages:
            if msg.hidden and msg.hidden.hidden_state is not None:
                states.append(msg.hidden.hidden_state)
        return states

    def combine_incoming_hidden(
        self,
        messages: list[Message],
    ) -> torch.Tensor | None:
        """Комбинировать скрытые состояния входящих сообщений."""
        states = self.extract_hidden_states(messages)
        if not states:
            return None
        return self.combine_hidden(states)

    def format_visible(
        self,
        messages: list[Message],
        agent_names: dict[str, str] | None = None,
    ) -> str:
        """Отформатировать видимые части сообщений для промпта."""
        parts = []
        for msg in messages:
            name = (agent_names or {}).get(msg.sender_id, msg.sender_id)
            parts.append(f"[{name}]:\n{msg.content}")
        return "\n\n".join(parts)
