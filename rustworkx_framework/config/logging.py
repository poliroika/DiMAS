import os
import sys

from loguru import logger as _logger

__all__ = ["logger", "setup_logging"]

logger = _logger

_DEFAULT_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
    "<level>{level: <8}</level> | "
    "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
    "<level>{message}</level>"
)


def _as_bool(value: str | None) -> bool:
    """
    Преобразовать строковое значение в булево.

    Интерпретирует `1/true/yes/on` (в любом регистре) как `True`, остальные
    значения или отсутствие переменной — как `False`.
    """
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def setup_logging(
    level: str | None = None,
    *,
    log_file: str | None = None,
    backtrace: bool | None = None,
    format_string: str | None = None,
) -> None:
    """
    Сконфигурировать loguru-логгер для консоли и, опционально, файла.

    Args:
        level: Уровень логирования (`DEBUG/INFO/WARNING/ERROR`). Если не указан,
            берётся из `RWXF_LOG_LEVEL` или `INFO` по умолчанию.
        log_file: Путь к файлу лога; при `None` можно задать через `RWXF_LOG_FILE`.
        backtrace: Включить детальный backtrace; по умолчанию читается из
            `RWXF_LOG_BACKTRACE` (`1/true/yes/on` => `True`).
        format_string: Формат сообщений; по умолчанию `RWXF_LOG_FORMAT` или `_DEFAULT_FORMAT`.

    Переменные окружения для файлового вывода:
        - `RWXF_LOG_ROTATION` (например, `10 MB`)
        - `RWXF_LOG_RETENTION` (например, `7 days`)
        - `RWXF_LOG_COMPRESSION` (например, `gz`)

    """
    configured_level = (level or os.getenv("RWXF_LOG_LEVEL") or "INFO").upper()
    configured_format = format_string or os.getenv("RWXF_LOG_FORMAT", _DEFAULT_FORMAT)
    configured_backtrace = backtrace if backtrace is not None else _as_bool(os.getenv("RWXF_LOG_BACKTRACE"))

    logger.remove()
    logger.add(
        sys.stderr,
        level=configured_level,
        format=configured_format,
        backtrace=configured_backtrace,
        diagnose=False,
        enqueue=True,
    )

    destination = log_file or os.getenv("RWXF_LOG_FILE")
    if destination:
        logger.add(
            destination,
            level=configured_level,
            format=configured_format,
            backtrace=configured_backtrace,
            diagnose=False,
            enqueue=True,
            rotation=os.getenv("RWXF_LOG_ROTATION", "10 MB"),
            retention=os.getenv("RWXF_LOG_RETENTION", "7 days"),
            compression=os.getenv("RWXF_LOG_COMPRESSION", "gz"),
        )
