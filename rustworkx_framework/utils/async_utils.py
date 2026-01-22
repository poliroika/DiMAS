import asyncio
from collections.abc import Awaitable, Coroutine
from typing import Any

__all__ = ["run_sync", "gather_with_concurrency"]


def run_sync[T](coro: Awaitable[T] | Coroutine[Any, Any, T], *, context: str = "run_sync") -> T:
    """Выполнить корутину синхронно, если нет активного event loop.

    Поднимает RuntimeError, если вызвано внутри уже запущенного цикла.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        if not asyncio.iscoroutine(coro):

            async def _wrap() -> T:
                return await coro  # type: ignore[misc]

            return asyncio.run(_wrap())
        return asyncio.run(coro)  # type: ignore[arg-type]

    raise RuntimeError(
        f"{context} cannot be called while an event loop is running. "
        "Use the asynchronous variant instead."
    )


async def gather_with_concurrency[T](
    n: int,
    *coros: Awaitable[T],
) -> list[T]:
    """Запустить корутины с ограничением числа одновременно выполняемых."""
    semaphore = asyncio.Semaphore(n)

    async def bounded(coro: Awaitable[T]) -> T:
        async with semaphore:
            return await coro

    return await asyncio.gather(*[bounded(c) for c in coros])


async def timeout_wrapper[T](
    coro: Awaitable[T],
    timeout: float,
    error_message: str = "Operation timed out",
) -> T:
    """Обернуть корутину таймаутом, пробрасывая TimeoutError с сообщением."""
    try:
        return await asyncio.wait_for(coro, timeout=timeout)
    except TimeoutError as exc:
        raise TimeoutError(error_message) from exc
