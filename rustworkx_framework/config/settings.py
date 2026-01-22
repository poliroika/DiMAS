import os
from pathlib import Path

from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

__all__ = ["FrameworkSettings", "load_env_file", "load_settings"]


class FrameworkSettings(BaseSettings):
    """Настройки фреймворка, загружаемые из окружения с префиксом `RWXF_`.

    Ключевые поля:
        - `api_key` / `api_key_file`: секретный ключ напрямую или путь к файлу.
        - `base_url`: базовый URL LLM-сервиса.
        - `model_name`: идентификатор модели генерации.
        - `embedding_model`: идентификатор модели эмбеддингов.
        - `log_*`: параметры логирования.
        - `default_timeout`, `max_retries`: сетевые таймауты и ретраи.
    """

    model_config = SettingsConfigDict(env_prefix="RWXF_", extra="ignore")

    api_key: SecretStr | None = Field(default=None, description="API key for LLM service")
    api_key_file: Path | None = Field(
        default=None,
        description="Path to a file that stores the API key securely",
    )
    base_url: str | None = Field(default=None, description="Base URL for LLM service")
    model_name: str = Field(default="gpt-4o-mini", description="LLM model identifier")
    embedding_model: str = Field(
        default="sentence-transformers/all-MiniLM-L6-v2",
        description="Embedding model identifier",
    )
    embedding_normalize: bool = Field(default=True, description="Normalize embeddings")
    embedding_fallback_dim: int = Field(default=384, description="Fallback dimension")
    log_level: str = Field(default="INFO", description="Logging level")
    log_file: str | None = Field(default=None, description="Log file path")
    log_backtrace: bool = Field(default=False, description="Enable backtrace")
    default_timeout: int = Field(default=60, description="Default timeout in seconds")
    max_retries: int = Field(default=3, description="Max retries for LLM calls")

    @field_validator("embedding_model")
    @classmethod
    def _validate_embedding_model(cls, value: str) -> str:
        """Проверить допустимость модели эмбеддингов."""
        if value == "hash" or value.startswith("hash:"):
            return value
        if value.startswith("sentence-transformers/") or value.startswith("sentence-transformers:"):
            return value
        raise ValueError(
            "Unsupported embedding model. Use 'sentence-transformers/<model>' or 'hash[:<dim>]'"
        )

    @field_validator("*", mode="before")
    @classmethod
    def _handle_empty_strings(cls, value):
        """Преобразовать пустые строки в None для корректной валидации."""
        if isinstance(value, str) and value.strip() == "":
            return None
        return value

    @field_validator("api_key_file")
    @classmethod
    def _validate_api_key_file(cls, value: Path | None) -> Path | None:
        """Убедиться, что файл с ключом существует перед чтением."""
        if value is None:
            return None
        if not value.is_file():
            raise ValueError(f"API key file not found: {value}")
        return value

    @model_validator(mode="after")
    def _load_secret_key(self) -> "FrameworkSettings":
        """Загрузить ключ из файла, если он не задан напрямую, и потребовать наличие."""
        if self.api_key is None and self.api_key_file is not None:
            content = self.api_key_file.read_text(encoding="utf-8").strip()
            if not content:
                raise ValueError("API key file is empty")
            object.__setattr__(self, "api_key", SecretStr(content))

        if self.api_key is None:
            raise ValueError("api_key is required via RWXF_API_KEY or RWXF_API_KEY_FILE")

        return self

    @property
    def resolved_api_key(self) -> str:
        """Вернуть секретное значение api_key или сгенерировать ошибку, если отсутствует."""
        if self.api_key is None:
            raise RuntimeError("API key is not configured")

        return self.api_key.get_secret_value()


def load_env_file(path: Path | str | None = None) -> None:
    """Загрузить переменные окружения из .env (если файл существует).

    Args:
        path: Путь к .env файлу; по умолчанию ищется в текущей директории.
    """
    env_path = Path(path or ".env")
    if not env_path.exists():
        return

    with env_path.open("r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" not in line:
                continue

            key, value = line.split("=", 1)
            key = key.strip()
            if not key:
                continue

            cleaned = value.strip().strip('"').strip("'")
            if key not in os.environ:
                os.environ[key] = cleaned


def load_settings(path: Path | str | None = None) -> FrameworkSettings:
    """Прочитать .env (если указан), загрузить и провалидировать настройки.

    Args:
        path: Путь к .env файлу для предварительной загрузки окружения.

    Returns:
        Провалидированный экземпляр `FrameworkSettings`.

    Raises:
        RuntimeError: если валидация настроек завершилась ошибкой.
    """
    load_env_file(path)

    try:
        settings = FrameworkSettings()
    except ValidationError as exc:
        messages = [err.get("msg", "invalid configuration value") for err in exc.errors()]
        detail = "; ".join(messages)
        raise RuntimeError(detail) from exc

    return settings
