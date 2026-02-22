"""
Web Search tool — поиск информации в интернете.

Позволяет агентам искать актуальную информацию в сети и читать содержимое веб-страниц.
Поддерживает DuckDuckGo (без API ключа), Serper (Google Search) и кастомные провайдеры.

Возможности:
- Поиск по запросу
- Автоматическое скачивание и парсинг страниц (fetch_content=True)
- Чтение конкретного URL (параметр url)
- Полноценный рендеринг JavaScript через Selenium (use_selenium=True)
"""

from __future__ import annotations

import contextlib
import html
import json
import logging
import re
import urllib.error
import urllib.parse
import urllib.request
from abc import ABC, abstractmethod
from collections import deque
from typing import Any, ClassVar, Self

from .base import BaseTool, ToolResult

logger = logging.getLogger(__name__)

# ============================================================
# HTML Parser — извлечение текста из HTML
# ============================================================


class SimpleHTMLParser:
    """
    Простой парсер HTML в текст без внешних зависимостей.

    Извлекает текстовое содержимое, удаляя теги, скрипты, стили.
    Для более качественного парсинга рекомендуется BeautifulSoup.
    """

    # Теги, содержимое которых нужно полностью удалить
    REMOVE_TAGS: ClassVar[set[str]] = {
        "script",
        "style",
        "head",
        "meta",
        "link",
        "noscript",
        "iframe",
        "svg",
        "nav",
        "footer",
        "header",
    }

    # Теги, после которых нужен перенос строки
    BLOCK_TAGS: ClassVar[set[str]] = {
        "p",
        "div",
        "br",
        "h1",
        "h2",
        "h3",
        "h4",
        "h5",
        "h6",
        "li",
        "tr",
        "article",
        "section",
    }

    @classmethod
    def html_to_text(cls, html_content: str, max_length: int = 8000) -> str:
        """
        Конвертировать HTML в чистый текст.

        Args:
            html_content: HTML строка.
            max_length: Максимальная длина результата.

        Returns:
            Извлечённый текст.

        """
        if not html_content:
            return ""

        text = html_content

        # Удаляем содержимое тегов script, style и т.д.
        for tag in cls.REMOVE_TAGS:
            pattern = rf"<{tag}[^>]*>.*?</{tag}>"
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE | re.DOTALL)

        # Удаляем HTML комментарии
        text = re.sub(r"<!--.*?-->", " ", text, flags=re.DOTALL)

        # Добавляем переносы строк после блочных тегов
        for tag in cls.BLOCK_TAGS:
            text = re.sub(rf"</{tag}>", f"</{tag}>\n", text, flags=re.IGNORECASE)
            text = re.sub(rf"<{tag}[^>]*/?>", f"\n<{tag}>", text, flags=re.IGNORECASE)

        # Заменяем <br> на перенос строки
        text = re.sub(r"<br\s*/?>", "\n", text, flags=re.IGNORECASE)

        # Удаляем все оставшиеся теги
        text = re.sub(r"<[^>]+>", " ", text)

        # Декодируем HTML entities
        text = html.unescape(text)

        # Нормализуем пробелы
        text = re.sub(r"[ \t]+", " ", text)  # Множественные пробелы в один
        text = re.sub(r"\n\s*\n+", "\n\n", text)  # Множественные переносы в два
        text = text.strip()

        # Ограничиваем длину
        if len(text) > max_length:
            text = text[:max_length] + "\n\n... (content truncated)"

        return text


# ============================================================
# URL Fetcher — скачивание и парсинг веб-страниц
# ============================================================


class URLFetcher:
    """Утилита для скачивания и парсинга веб-страниц."""

    DEFAULT_HEADERS: ClassVar[dict[str, str]] = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.5",
    }

    def __init__(self, timeout: int = 15, max_content_length: int = 500_000):
        """
        Создать URLFetcher.

        Args:
            timeout: Таймаут запроса в секундах.
            max_content_length: Максимальный размер скачиваемого контента в байтах.

        """
        self._timeout = timeout
        self._max_content_length = max_content_length

    def fetch(self, url: str) -> dict[str, Any]:
        """
        Скачать и распарсить веб-страницу.

        Args:
            url: URL страницы.

        Returns:
            Словарь с ключами:
            - success: bool
            - url: str
            - title: str (если найден)
            - content: str (текстовое содержимое)
            - error: str (если ошибка)

        """
        result: dict[str, Any] = {
            "success": False,
            "url": url,
            "title": "",
            "content": "",
            "error": "",
        }

        try:
            # Создаём запрос
            request = urllib.request.Request(url, headers=self.DEFAULT_HEADERS)

            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                # Проверяем Content-Type
                content_type = response.headers.get("Content-Type", "")
                if "text/html" not in content_type and "text/plain" not in content_type:
                    result["error"] = f"Unsupported content type: {content_type}"
                    return result

                # Читаем контент с ограничением
                raw_content = response.read(self._max_content_length)

                # Определяем кодировку
                charset = "utf-8"
                if "charset=" in content_type:
                    match = re.search(r"charset=([^\s;]+)", content_type)
                    if match:
                        charset = match.group(1)

                try:
                    html_content = raw_content.decode(charset, errors="replace")
                except (UnicodeDecodeError, LookupError):
                    html_content = raw_content.decode("utf-8", errors="replace")

                # Извлекаем title
                title_match = re.search(r"<title[^>]*>(.*?)</title>", html_content, re.IGNORECASE | re.DOTALL)
                if title_match:
                    result["title"] = html.unescape(title_match.group(1).strip())

                # Извлекаем основной контент (пробуем найти main/article)
                main_content = html_content
                for tag in ["main", "article", "div[role='main']"]:
                    match = re.search(
                        rf"<{tag}[^>]*>(.*?)</{tag.split('[')[0]}>",
                        html_content,
                        re.IGNORECASE | re.DOTALL,
                    )
                    # Minimum content length to be considered main content
                    min_main_content_length = 500
                    if match and len(match.group(1)) > min_main_content_length:
                        main_content = match.group(1)
                        break

                # Парсим HTML в текст
                result["content"] = SimpleHTMLParser.html_to_text(main_content)
                result["success"] = True

        except urllib.error.HTTPError as e:
            result["error"] = f"HTTP Error {e.code}: {e.reason}"
        except urllib.error.URLError as e:
            result["error"] = f"URL Error: {e.reason}"
        except TimeoutError:
            result["error"] = f"Request timed out after {self._timeout} seconds"
        except (ValueError, OSError, UnicodeDecodeError) as e:
            result["error"] = f"Fetch error: {e}"

        return result


# ============================================================
# Selenium Fetcher — полноценный рендеринг через браузер
# ============================================================


class SeleniumFetcher:
    """
    Fetcher на базе Selenium WebDriver для полноценного рендеринга страниц.

    Позволяет:
    - Рендерить JavaScript (SPA, динамический контент)
    - Ждать загрузки элементов
    - Скроллить страницу для подгрузки контента
    - Обрабатывать сайты, которые блокируют простые HTTP-запросы

    Требует установки: ``pip install selenium webdriver-manager``

    Example:
        # Базовое использование
        fetcher = SeleniumFetcher()
        result = fetcher.fetch("https://example.com")
        print(result["content"])

        # С настройками
        fetcher = SeleniumFetcher(
            headless=True,
            wait_timeout=15,
            scroll_to_bottom=True,
            browser="chrome",
        )
        result = fetcher.fetch("https://spa-website.com")

    """

    def __init__(
        self,
        *,
        headless: bool = True,
        browser: str = "chrome",
        wait_timeout: int = 15,
        page_load_timeout: int = 30,
        max_content_length: int = 500_000,
        scroll_to_bottom: bool = False,
        scroll_pause: float = 1.0,
        max_scrolls: int = 5,
        extra_wait: float = 2.0,
        user_agent: str | None = None,
        window_size: tuple[int, int] = (1920, 1080),
        proxy: str | None = None,
        disable_images: bool = False,
    ):
        """
        Создать SeleniumFetcher.

        Args:
            headless: Запускать браузер без GUI (рекомендуется для серверов).
            browser: Тип браузера: "chrome", "firefox", "edge".
            wait_timeout: Таймаут ожидания элементов (сек).
            page_load_timeout: Таймаут загрузки страницы (сек).
            max_content_length: Максимальная длина извлечённого контента.
            scroll_to_bottom: Скроллить страницу до конца для подгрузки контента.
            scroll_pause: Пауза между скроллами (сек).
            max_scrolls: Максимальное количество скроллов.
            extra_wait: Дополнительное ожидание после загрузки (сек).
            user_agent: Кастомный User-Agent (None = стандартный).
            window_size: Размер окна браузера (ширина, высота).
            proxy: Прокси-сервер (например "http://proxy:8080").
            disable_images: Отключить загрузку изображений (быстрее).

        """
        self._headless = headless
        self._browser = browser.lower()
        self._wait_timeout = wait_timeout
        self._page_load_timeout = page_load_timeout
        self._max_content_length = max_content_length
        self._scroll_to_bottom = scroll_to_bottom
        self._scroll_pause = scroll_pause
        self._max_scrolls = max_scrolls
        self._extra_wait = extra_wait
        self._user_agent = user_agent
        self._window_size = window_size
        self._proxy = proxy
        self._disable_images = disable_images

        # Lazy-initialized driver
        self._driver: Any = None

    def _ensure_dependencies(self) -> None:
        """Проверить наличие selenium и webdriver-manager."""
        try:
            import selenium  # noqa: F401
        except ImportError as e:
            msg = "Selenium is required for SeleniumFetcher. Install it with: pip install selenium webdriver-manager"
            raise ImportError(msg) from e

    def _create_driver(self) -> Any:
        """Создать и настроить WebDriver."""
        self._ensure_dependencies()

        if self._browser == "chrome":
            return self._create_chrome_driver()
        if self._browser == "firefox":
            return self._create_firefox_driver()
        if self._browser == "edge":
            return self._create_edge_driver()

        msg = f"Unsupported browser: {self._browser}. Use 'chrome', 'firefox', or 'edge'."
        raise ValueError(msg)

    def _create_chrome_driver(self) -> Any:
        """Создать Chrome WebDriver."""
        from selenium import webdriver
        from selenium.webdriver.chrome.options import Options
        from selenium.webdriver.chrome.service import Service

        options = Options()
        if self._headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument("--disable-gpu")
        options.add_argument(f"--window-size={self._window_size[0]},{self._window_size[1]}")
        options.add_argument("--disable-blink-features=AutomationControlled")

        if self._user_agent:
            options.add_argument(f"--user-agent={self._user_agent}")
        if self._proxy:
            options.add_argument(f"--proxy-server={self._proxy}")
        if self._disable_images:
            prefs = {"profile.managed_default_content_settings.images": 2}
            options.add_experimental_option("prefs", prefs)

        # Попробуем webdriver-manager, иначе системный chromedriver
        try:
            from webdriver_manager.chrome import ChromeDriverManager

            service = Service(ChromeDriverManager().install())
        except Exception:  # noqa: BLE001
            logger.debug("webdriver-manager failed or not found, using system chromedriver")
            service = Service()

        return webdriver.Chrome(service=service, options=options)

    def _create_firefox_driver(self) -> Any:
        """Создать Firefox WebDriver."""
        from selenium import webdriver
        from selenium.webdriver.firefox.options import Options
        from selenium.webdriver.firefox.service import Service

        options = Options()
        if self._headless:
            options.add_argument("--headless")
        options.set_preference("general.useragent.override", self._user_agent or "")
        if self._proxy:
            # Firefox proxy через preferences
            from urllib.parse import urlparse

            parsed = urlparse(self._proxy)
            options.set_preference("network.proxy.type", 1)
            options.set_preference("network.proxy.http", parsed.hostname)
            options.set_preference("network.proxy.http_port", parsed.port or 8080)
        if self._disable_images:
            options.set_preference("permissions.default.image", 2)

        try:
            from webdriver_manager.firefox import GeckoDriverManager

            service = Service(GeckoDriverManager().install())
        except Exception:  # noqa: BLE001
            logger.debug("webdriver-manager failed or not found, using system geckodriver")
            service = Service()

        return webdriver.Firefox(service=service, options=options)

    def _create_edge_driver(self) -> Any:
        """Создать Edge WebDriver."""
        from selenium import webdriver
        from selenium.webdriver.edge.options import Options
        from selenium.webdriver.edge.service import Service

        options = Options()
        if self._headless:
            options.add_argument("--headless=new")
        options.add_argument("--no-sandbox")
        options.add_argument("--disable-dev-shm-usage")
        options.add_argument(f"--window-size={self._window_size[0]},{self._window_size[1]}")

        if self._user_agent:
            options.add_argument(f"--user-agent={self._user_agent}")
        if self._proxy:
            options.add_argument(f"--proxy-server={self._proxy}")

        try:
            from webdriver_manager.microsoft import EdgeChromiumDriverManager

            service = Service(EdgeChromiumDriverManager().install())
        except Exception:  # noqa: BLE001
            logger.debug("webdriver-manager failed or not found, using system msedgedriver")
            service = Service()

        return webdriver.Edge(service=service, options=options)

    def _get_driver(self) -> Any:
        """Получить или создать WebDriver (lazy init)."""
        if self._driver is None:
            self._driver = self._create_driver()
            self._driver.set_page_load_timeout(self._page_load_timeout)
            self._driver.implicitly_wait(self._wait_timeout)
        return self._driver

    def _scroll_page(self, driver: Any) -> None:
        """Скроллить страницу для подгрузки динамического контента."""
        import time

        last_height = driver.execute_script("return document.body.scrollHeight")

        for _ in range(self._max_scrolls):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(self._scroll_pause)

            new_height = driver.execute_script("return document.body.scrollHeight")
            if new_height == last_height:
                break
            last_height = new_height

    def fetch(self, url: str) -> dict[str, Any]:
        """
        Открыть URL в браузере, дождаться рендеринга и извлечь контент.

        Args:
            url: URL страницы.

        Returns:
            Словарь с ключами:
            - success: bool
            - url: str
            - title: str
            - content: str (текстовое содержимое)
            - error: str (если ошибка)

        """
        import time

        result: dict[str, Any] = {
            "success": False,
            "url": url,
            "title": "",
            "content": "",
            "error": "",
        }

        try:
            driver = self._get_driver()
            driver.get(url)

            # Дополнительное ожидание для рендеринга JS
            if self._extra_wait > 0:
                time.sleep(self._extra_wait)

            # Скроллинг для подгрузки контента
            if self._scroll_to_bottom:
                self._scroll_page(driver)

            # Извлекаем title
            result["title"] = driver.title or ""

            # Извлекаем HTML после рендеринга
            html_content = driver.page_source

            # Пробуем найти основной контент
            main_content = html_content
            for tag in ["main", "article"]:
                match = re.search(
                    rf"<{tag}[^>]*>(.*?)</{tag}>",
                    html_content,
                    re.IGNORECASE | re.DOTALL,
                )
                min_main_content_length = 500
                if match and len(match.group(1)) > min_main_content_length:
                    main_content = match.group(1)
                    break

            # Парсим HTML в текст
            content = SimpleHTMLParser.html_to_text(main_content, max_length=self._max_content_length)
            result["content"] = content
            result["success"] = True

        except Exception as e:  # noqa: BLE001
            error_type = type(e).__name__
            result["error"] = f"Selenium error ({error_type}): {e}"
            logger.debug("SeleniumFetcher error for %s: %s", url, result["error"])

        return result

    def fetch_with_wait(
        self,
        url: str,
        wait_for_selector: str | None = None,
        wait_timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Открыть URL и дождаться появления конкретного CSS-селектора.

        Args:
            url: URL страницы.
            wait_for_selector: CSS-селектор элемента, которого нужно дождаться.
            wait_timeout: Таймаут ожидания (сек). По умолчанию self._wait_timeout.

        Returns:
            Словарь с результатом (как в fetch).

        """
        import time

        result: dict[str, Any] = {
            "success": False,
            "url": url,
            "title": "",
            "content": "",
            "error": "",
        }

        try:
            driver = self._get_driver()
            driver.get(url)

            if wait_for_selector:
                from selenium.webdriver.common.by import By
                from selenium.webdriver.support import expected_conditions as EC  # noqa: N812
                from selenium.webdriver.support.ui import WebDriverWait

                timeout = wait_timeout or self._wait_timeout
                WebDriverWait(driver, timeout).until(
                    EC.presence_of_element_located((By.CSS_SELECTOR, wait_for_selector))
                )
            elif self._extra_wait > 0:
                time.sleep(self._extra_wait)

            if self._scroll_to_bottom:
                self._scroll_page(driver)

            result["title"] = driver.title or ""
            html_content = driver.page_source

            main_content = html_content
            for tag in ["main", "article"]:
                match = re.search(
                    rf"<{tag}[^>]*>(.*?)</{tag}>",
                    html_content,
                    re.IGNORECASE | re.DOTALL,
                )
                min_main_content_length = 500
                if match and len(match.group(1)) > min_main_content_length:
                    main_content = match.group(1)
                    break

            content = SimpleHTMLParser.html_to_text(main_content, max_length=self._max_content_length)
            result["content"] = content
            result["success"] = True

        except Exception as e:  # noqa: BLE001
            error_type = type(e).__name__
            result["error"] = f"Selenium error ({error_type}): {e}"
            logger.debug("SeleniumFetcher.fetch_with_wait error for %s: %s", url, result["error"])

        return result

    # ================================================================
    # Browser Actions — взаимодействие со страницей
    # ================================================================

    def click_element(self, selector: str, wait_timeout: int | None = None) -> dict[str, Any]:
        """
        Кликнуть по элементу на текущей странице.

        Args:
            selector: CSS-селектор элемента.
            wait_timeout: Таймаут ожидания элемента (сек).

        Returns:
            Словарь с результатом:
            - success: bool
            - url: str (текущий URL после клика)
            - title: str
            - clicked_text: str (текст элемента, по которому кликнули)
            - error: str

        """
        result: dict[str, Any] = {
            "success": False,
            "url": "",
            "title": "",
            "clicked_text": "",
            "error": "",
        }

        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.support import expected_conditions as EC  # noqa: N812
            from selenium.webdriver.support.ui import WebDriverWait

            driver = self._get_driver()
            timeout = wait_timeout or self._wait_timeout

            element = WebDriverWait(driver, timeout).until(EC.element_to_be_clickable((By.CSS_SELECTOR, selector)))
            result["clicked_text"] = element.text or element.get_attribute("textContent") or ""
            element.click()

            import time

            time.sleep(1.0)  # Ждём навигацию/рендеринг

            result["url"] = driver.current_url
            result["title"] = driver.title or ""
            result["success"] = True

        except Exception as e:  # noqa: BLE001
            result["error"] = f"Click error ({type(e).__name__}): {e}"

        return result

    def fill_input(
        self,
        selector: str,
        value: str,
        *,
        submit: bool = False,
        clear_first: bool = True,
        wait_timeout: int | None = None,
    ) -> dict[str, Any]:
        """
        Заполнить поле ввода на текущей странице.

        Args:
            selector: CSS-селектор поля ввода.
            value: Значение для ввода.
            submit: Нажать Enter после ввода.
            clear_first: Очистить поле перед вводом.
            wait_timeout: Таймаут ожидания элемента (сек).

        Returns:
            Словарь с результатом:
            - success: bool
            - url: str
            - title: str
            - error: str

        """
        result: dict[str, Any] = {
            "success": False,
            "url": "",
            "title": "",
            "error": "",
        }

        try:
            from selenium.webdriver.common.by import By
            from selenium.webdriver.common.keys import Keys
            from selenium.webdriver.support import expected_conditions as EC  # noqa: N812
            from selenium.webdriver.support.ui import WebDriverWait

            driver = self._get_driver()
            timeout = wait_timeout or self._wait_timeout

            element = WebDriverWait(driver, timeout).until(EC.presence_of_element_located((By.CSS_SELECTOR, selector)))

            if clear_first:
                element.clear()

            element.send_keys(value)

            if submit:
                element.send_keys(Keys.RETURN)
                import time

                time.sleep(2.0)  # Ждём отправку формы

            result["url"] = driver.current_url
            result["title"] = driver.title or ""
            result["success"] = True

        except Exception as e:  # noqa: BLE001
            result["error"] = f"Fill error ({type(e).__name__}): {e}"

        return result

    def extract_links(
        self,
        selector: str = "a[href]",
        *,
        base_url_filter: str | None = None,
        max_links: int = 50,
    ) -> dict[str, Any]:
        """
        Извлечь все ссылки с текущей страницы.

        Args:
            selector: CSS-селектор для поиска ссылок.
            base_url_filter: Фильтр по базовому URL (только ссылки начинающиеся с этого URL).
            max_links: Максимальное количество ссылок.

        Returns:
            Словарь с результатом:
            - success: bool
            - url: str (текущий URL)
            - links: list[dict] — список ссылок [{url, text, title}]
            - count: int
            - error: str

        """
        result: dict[str, Any] = {
            "success": False,
            "url": "",
            "links": [],
            "count": 0,
            "error": "",
        }

        try:
            from selenium.webdriver.common.by import By

            driver = self._get_driver()
            result["url"] = driver.current_url

            elements = driver.find_elements(By.CSS_SELECTOR, selector)
            links: list[dict[str, str]] = []

            for elem in elements:
                href = elem.get_attribute("href") or ""
                if not href or href.startswith(("javascript:", "mailto:", "tel:", "#")):
                    continue

                if base_url_filter and not href.startswith(base_url_filter):
                    continue

                links.append(
                    {
                        "url": href,
                        "text": (elem.text or "").strip()[:200],
                        "title": (elem.get_attribute("title") or "").strip()[:200],
                    }
                )

                if len(links) >= max_links:
                    break

            result["links"] = links
            result["count"] = len(links)
            result["success"] = True

        except Exception as e:  # noqa: BLE001
            result["error"] = f"Extract links error ({type(e).__name__}): {e}"

        return result

    def execute_js(self, script: str) -> dict[str, Any]:
        """
        Выполнить произвольный JavaScript на текущей странице.

        Args:
            script: JavaScript код для выполнения.

        Returns:
            Словарь с результатом:
            - success: bool
            - url: str
            - return_value: Any (результат выполнения скрипта)
            - error: str

        """
        result: dict[str, Any] = {
            "success": False,
            "url": "",
            "return_value": None,
            "error": "",
        }

        try:
            driver = self._get_driver()
            return_value = driver.execute_script(script)
            result["url"] = driver.current_url
            result["return_value"] = str(return_value) if return_value is not None else None
            result["success"] = True

        except Exception as e:  # noqa: BLE001
            result["error"] = f"JS execution error ({type(e).__name__}): {e}"

        return result

    def get_current_url(self) -> str:
        """Получить текущий URL открытой страницы."""
        try:
            driver = self._get_driver()
        except Exception:  # noqa: BLE001
            return ""
        else:
            return driver.current_url

    def get_page_content(self) -> dict[str, Any]:
        """
        Извлечь текстовое содержимое текущей открытой страницы.

        Returns:
            Словарь с ключами: success, url, title, content, error.

        """
        result: dict[str, Any] = {
            "success": False,
            "url": "",
            "title": "",
            "content": "",
            "error": "",
        }

        try:
            driver = self._get_driver()
            result["url"] = driver.current_url
            result["title"] = driver.title or ""

            html_content = driver.page_source
            main_content = html_content
            for tag in ["main", "article"]:
                match = re.search(
                    rf"<{tag}[^>]*>(.*?)</{tag}>",
                    html_content,
                    re.IGNORECASE | re.DOTALL,
                )
                min_main_content_length = 500
                if match and len(match.group(1)) > min_main_content_length:
                    main_content = match.group(1)
                    break

            result["content"] = SimpleHTMLParser.html_to_text(main_content, max_length=self._max_content_length)
            result["success"] = True

        except Exception as e:  # noqa: BLE001
            result["error"] = f"Get content error ({type(e).__name__}): {e}"

        return result

    def crawl(
        self,
        start_url: str,
        *,
        max_pages: int = 10,
        max_depth: int = 2,
        url_filter: str | None = None,
        link_selector: str = "a[href]",
        extract_content: bool = True,
    ) -> dict[str, Any]:
        """
        Рекурсивно обойти сайт, собирая контент со страниц.

        Args:
            start_url: Начальный URL.
            max_pages: Максимальное количество страниц для обхода.
            max_depth: Максимальная глубина обхода.
            url_filter: Фильтр URL (только ссылки содержащие эту подстроку).
                       По умолчанию — тот же домен что и start_url.
            link_selector: CSS-селектор для поиска ссылок.
            extract_content: Извлекать текстовое содержимое страниц.

        Returns:
            Словарь с результатом:
            - success: bool
            - pages: list[dict] — [{url, title, content, depth, links_found}]
            - total_pages: int
            - error: str

        """
        import time
        from urllib.parse import urlparse

        result: dict[str, Any] = {
            "success": False,
            "pages": [],
            "total_pages": 0,
            "error": "",
        }

        # Определяем фильтр по домену
        if url_filter is None:
            parsed = urlparse(start_url)
            url_filter = f"{parsed.scheme}://{parsed.netloc}"

        visited: set[str] = set()
        queue: deque[tuple[str, int]] = deque([(start_url, 0)])  # (url, depth)
        pages: list[dict[str, Any]] = []

        try:
            while queue and len(pages) < max_pages:
                current_url, depth = queue.popleft()

                # Нормализуем URL (убираем фрагменты)
                current_url = current_url.split("#")[0].rstrip("/")
                if current_url in visited:
                    continue
                visited.add(current_url)

                # Открываем страницу
                fetch_result = self.fetch(current_url)
                if not fetch_result["success"]:
                    continue

                page_info: dict[str, Any] = {
                    "url": current_url,
                    "title": fetch_result["title"],
                    "depth": depth,
                    "links_found": 0,
                }

                if extract_content:
                    page_info["content"] = fetch_result["content"]

                # Извлекаем ссылки если не достигли максимальной глубины
                if depth < max_depth:
                    links_result = self.extract_links(
                        selector=link_selector,
                        base_url_filter=url_filter,
                        max_links=50,
                    )
                    if links_result["success"]:
                        page_info["links_found"] = links_result["count"]
                        for link in links_result["links"]:
                            link_url = link["url"].split("#")[0].rstrip("/")
                            if link_url not in visited and len(queue) < max_pages * 2:
                                queue.append((link_url, depth + 1))

                pages.append(page_info)
                time.sleep(0.5)  # Вежливая пауза между запросами

            result["pages"] = pages
            result["total_pages"] = len(pages)
            result["success"] = True

        except Exception as e:  # noqa: BLE001
            result["error"] = f"Crawl error ({type(e).__name__}): {e}"
            result["pages"] = pages
            result["total_pages"] = len(pages)

        return result

    def close(self) -> None:
        """Закрыть браузер и освободить ресурсы."""
        if self._driver is not None:
            with contextlib.suppress(Exception):
                self._driver.quit()
            self._driver = None

    def __del__(self) -> None:
        """Закрыть браузер при удалении объекта."""
        self.close()

    def __enter__(self) -> Self:
        """Поддержка context manager."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Закрыть браузер при выходе из context manager."""
        self.close()


# ============================================================
# Search Providers
# ============================================================


class SearchProvider(ABC):
    """Абстрактный базовый класс для провайдеров поиска."""

    @abstractmethod
    def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """
        Выполнить поиск и вернуть результаты.

        Args:
            query: Поисковый запрос.
            max_results: Максимальное количество результатов.

        Returns:
            Список словарей с ключами 'title', 'url', 'snippet'.

        """
        ...


class DuckDuckGoProvider(SearchProvider):
    """
    Провайдер поиска через DuckDuckGo Instant Answers API.

    Не требует API ключа. Использует публичный API DuckDuckGo.
    Ограничение: возвращает только instant answers и related topics,
    а не полные результаты поиска (это ограничение API).

    Для полноценного поиска рекомендуется использовать Serper или TavilyProvider.
    """

    def __init__(self, timeout: int = 10):
        """
        Создать DuckDuckGoProvider.

        Args:
            timeout: Таймаут запроса в секундах.

        """
        self._timeout = timeout
        self._base_url = "https://api.duckduckgo.com/"

    def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """
        Выполнить поиск через DuckDuckGo.

        Args:
            query: Поисковый запрос.
            max_results: Максимальное количество результатов.

        Returns:
            Список результатов поиска.

        """
        results: list[dict[str, str]] = []

        try:
            # Формируем URL для API запроса
            params = urllib.parse.urlencode(
                {
                    "q": query,
                    "format": "json",
                    "no_html": "1",
                    "skip_disambig": "1",
                }
            )
            url = f"{self._base_url}?{params}"

            # Выполняем запрос
            request = urllib.request.Request(
                url,
                headers={"User-Agent": "RustworkxFramework/1.0"},
            )
            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                data = json.loads(response.read().decode("utf-8"))

            # Извлекаем Abstract (основной ответ)
            if data.get("Abstract"):
                results.append(
                    {
                        "title": data.get("Heading", "DuckDuckGo Answer"),
                        "url": data.get("AbstractURL", ""),
                        "snippet": data.get("Abstract", ""),
                    }
                )

            # Извлекаем Related Topics
            results.extend(
                {
                    "title": topic.get("Text", "")[:80],
                    "url": topic.get("FirstURL", ""),
                    "snippet": topic.get("Text", ""),
                }
                for topic in data.get("RelatedTopics", [])[: max_results - len(results)]
                if isinstance(topic, dict) and topic.get("Text")
            )

            # Если есть Results (редко для обычных запросов)
            results.extend(
                {
                    "title": item.get("Text", "")[:80],
                    "url": item.get("FirstURL", ""),
                    "snippet": item.get("Text", ""),
                }
                for item in data.get("Results", [])[: max_results - len(results)]
                if isinstance(item, dict)
            )

        except (urllib.error.URLError, ValueError, KeyError, OSError):
            # Ошибки сети обрабатываются в WebSearchTool
            pass

        return results[:max_results]


class SerperProvider(SearchProvider):
    """
    Провайдер поиска через Serper API (Google Search).

    Требует API ключ от https://serper.dev/
    Возвращает полноценные результаты Google Search.
    """

    def __init__(self, api_key: str, timeout: int = 10):
        """
        Создать SerperProvider.

        Args:
            api_key: API ключ Serper.
            timeout: Таймаут запроса в секундах.

        """
        self._api_key = api_key
        self._timeout = timeout
        self._base_url = "https://google.serper.dev/search"

    def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """
        Выполнить поиск через Serper (Google).

        Args:
            query: Поисковый запрос.
            max_results: Максимальное количество результатов.

        Returns:
            Список результатов поиска.

        """
        results: list[dict[str, str]] = []

        try:
            # Формируем запрос
            payload = json.dumps({"q": query, "num": max_results}).encode("utf-8")
            request = urllib.request.Request(
                self._base_url,
                data=payload,
                headers={
                    "X-API-KEY": self._api_key,
                    "Content-Type": "application/json",
                },
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                data = json.loads(response.read().decode("utf-8"))

            # Извлекаем organic results
            organic_items = data.get("organic", [])[:max_results]
            results.extend(
                {
                    "title": item.get("title", ""),
                    "url": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                }
                for item in organic_items
            )

            # Добавляем answer box если есть
            if data.get("answerBox") and len(results) < max_results:
                answer = data["answerBox"]
                results.insert(
                    0,
                    {
                        "title": answer.get("title", "Featured Answer"),
                        "url": answer.get("link", ""),
                        "snippet": answer.get("snippet", answer.get("answer", "")),
                    },
                )

        except (urllib.error.URLError, ValueError, KeyError, OSError):
            pass

        return results[:max_results]


class TavilyProvider(SearchProvider):
    """
    Провайдер поиска через Tavily API (как в LangGraph).

    Требует API ключ от https://tavily.com/
    Возвращает результаты с возможностью получения полного содержимого страниц.
    """

    def __init__(
        self,
        api_key: str,
        timeout: int = 30,
        *,
        include_answer: bool = True,
        search_depth: str = "basic",  # "basic" or "advanced"
    ):
        """
        Создать TavilyProvider.

        Args:
            api_key: API ключ Tavily.
            timeout: Таймаут запроса в секундах.
            include_answer: Включить AI-генерированный ответ.
            search_depth: Глубина поиска ("basic" или "advanced").

        """
        self._api_key = api_key
        self._timeout = timeout
        self._include_answer = include_answer
        self._search_depth = search_depth
        self._base_url = "https://api.tavily.com/search"

    def search(self, query: str, max_results: int = 5) -> list[dict[str, str]]:
        """
        Выполнить поиск через Tavily.

        Args:
            query: Поисковый запрос.
            max_results: Максимальное количество результатов.

        Returns:
            Список результатов поиска.

        """
        results: list[dict[str, str]] = []

        try:
            payload = json.dumps(
                {
                    "api_key": self._api_key,
                    "query": query,
                    "max_results": max_results,
                    "include_answer": self._include_answer,
                    "search_depth": self._search_depth,
                }
            ).encode("utf-8")

            request = urllib.request.Request(
                self._base_url,
                data=payload,
                headers={"Content-Type": "application/json"},
                method="POST",
            )

            with urllib.request.urlopen(request, timeout=self._timeout) as response:
                data = json.loads(response.read().decode("utf-8"))

            # Добавляем AI-ответ если есть
            if data.get("answer"):
                results.append(
                    {
                        "title": "Tavily AI Answer",
                        "url": "",
                        "snippet": data["answer"],
                    }
                )

            # Извлекаем результаты
            results.extend(
                {
                    "title": item.get("title", ""),
                    "url": item.get("url", ""),
                    "snippet": item.get("content", ""),
                }
                for item in data.get("results", [])[:max_results]
            )

        except (urllib.error.URLError, ValueError, KeyError, OSError):
            pass

        return results[:max_results]


# ============================================================
# WebSearchTool — основной инструмент
# ============================================================


class WebSearchTool(BaseTool):
    """
    Инструмент для поиска информации в интернете с возможностью чтения страниц.

    Позволяет агентам:
    1. Искать информацию по запросу (action="search")
    2. Читать конкретный URL (action="fetch")
    3. Кликать по элементам на странице (action="click", требует Selenium)
    4. Заполнять формы (action="fill", требует Selenium)
    5. Извлекать ссылки со страницы (action="extract_links", требует Selenium)
    6. Выполнять JavaScript (action="execute_js", требует Selenium)
    7. Рекурсивно обходить сайт (action="crawl", требует Selenium)
    8. Получить содержимое текущей страницы (action="get_content", требует Selenium)

    Все события логируются через callback-систему фреймворка (on_tool_start/end/error).

    Example:
        # Базовое использование — только поиск (быстро)
        tool = WebSearchTool()
        result = tool.execute(query="Python async await tutorial")

        # С автоматическим скачиванием страниц
        tool = WebSearchTool(fetch_content=True)
        result = tool.execute(query="Python asyncio best practices")

        # Чтение конкретного URL
        result = tool.execute(url="https://docs.python.org/3/library/asyncio.html")

        # С Selenium — полноценный браузер
        tool = WebSearchTool(
            use_selenium=True,
            fetch_content=True,
            selenium_config={"headless": True, "browser": "chrome"},
        )

        # Клик по элементу
        result = tool.execute(action="click", selector="button.submit")

        # Заполнение формы
        result = tool.execute(action="fill", selector="input[name=q]", value="Python", submit=True)

        # Извлечение ссылок
        result = tool.execute(action="extract_links", url="https://example.com")

        # Выполнение JavaScript
        result = tool.execute(action="execute_js", js_code="return document.title")

        # Рекурсивный обход сайта
        result = tool.execute(action="crawl", url="https://docs.python.org", max_depth=2, max_pages=5)

    """

    def __init__(
        self,
        provider: SearchProvider | None = None,
        max_results: int = 5,
        max_content_length: int = 4000,
        *,
        fetch_content: bool = False,
        timeout: int = 15,
        use_selenium: bool = False,
        selenium_config: dict[str, Any] | None = None,
        selenium_fetcher: SeleniumFetcher | None = None,
        callback_manager: Any | None = None,
    ):
        """
        Создать WebSearchTool.

        Args:
            provider: Провайдер поиска (по умолчанию DuckDuckGoProvider).
            max_results: Максимальное количество результатов поиска.
            max_content_length: Максимальная длина контента каждой страницы.
            fetch_content: Автоматически скачивать содержимое страниц при поиске.
            timeout: Таймаут запроса в секундах.
            use_selenium: Использовать Selenium для скачивания страниц.
            selenium_config: Настройки для SeleniumFetcher (если use_selenium=True).
            selenium_fetcher: Готовый экземпляр SeleniumFetcher.
            callback_manager: CallbackManager для отправки событий.
                             Если None — пытается получить из контекста.

        """
        self._provider = provider or DuckDuckGoProvider(timeout=timeout)
        self._max_results = max_results
        self._max_content_length = max_content_length
        self._fetch_content = fetch_content
        self._timeout = timeout
        self._fetcher = URLFetcher(timeout=timeout, max_content_length=500_000)
        self._callback_manager = callback_manager

        # Selenium support
        self._use_selenium = use_selenium
        self._selenium_fetcher: SeleniumFetcher | None = None

        if selenium_fetcher is not None:
            self._selenium_fetcher = selenium_fetcher
            self._use_selenium = True
        elif use_selenium:
            config = selenium_config or {}
            self._selenium_fetcher = SeleniumFetcher(**config)

    def _get_callback_manager(self) -> Any | None:
        """Получить callback manager (из конструктора или из контекста)."""
        if self._callback_manager is not None:
            return self._callback_manager
        try:
            from ..callbacks.context import get_callback_manager

            return get_callback_manager()
        except Exception:  # noqa: BLE001
            return None

    def _emit_tool_start(self, action: str, arguments: dict[str, Any] | None = None) -> None:
        """Отправить событие начала выполнения tool."""
        from uuid import uuid4

        cb = self._get_callback_manager()
        if cb is not None:
            with contextlib.suppress(Exception):
                cb.on_tool_start(
                    uuid4(),
                    tool_name=self.name,
                    action=action,
                    arguments=arguments or {},
                )

    def _emit_tool_end(
        self,
        action: str,
        *,
        success: bool = True,
        output_size: int = 0,
        duration_ms: float = 0.0,
        result_summary: str = "",
    ) -> None:
        """Отправить событие завершения tool."""
        from uuid import uuid4

        cb = self._get_callback_manager()
        if cb is not None:
            with contextlib.suppress(Exception):
                cb.on_tool_end(
                    uuid4(),
                    tool_name=self.name,
                    action=action,
                    success=success,
                    output_size=output_size,
                    duration_ms=duration_ms,
                    result_summary=result_summary,
                )

    def _emit_tool_error(self, action: str, error: Exception) -> None:
        """Отправить событие ошибки tool."""
        from uuid import uuid4

        cb = self._get_callback_manager()
        if cb is not None:
            with contextlib.suppress(Exception):
                cb.on_tool_error(
                    uuid4(),
                    tool_name=self.name,
                    action=action,
                    error_type=type(error).__name__,
                    error_message=str(error),
                )

    @property
    def name(self) -> str:
        return "web_search"

    @property
    def description(self) -> str:
        base = (
            "Search the web and interact with web pages. "
            "Use 'query' to search for information. "
            "Use 'url' to read a specific web page. "
            "Set 'fetch_content=true' to automatically read the content of search results. "
            "Returns search results with titles, URLs, snippets, and optionally full page content."
        )
        if self._use_selenium:
            base += (
                "\n\nThis tool uses a real browser (Selenium) and supports advanced actions:\n"
                "- action='click': Click an element by CSS selector.\n"
                "- action='fill': Fill an input field and optionally submit.\n"
                "- action='extract_links': Extract all links from the current/specified page.\n"
                "- action='execute_js': Execute JavaScript code on the current page.\n"
                "- action='crawl': Recursively crawl a website collecting content.\n"
                "- action='get_content': Get text content of the current page.\n"
                "Use 'wait_for_selector' to wait for a specific element before extracting content."
            )
        return base

    @property
    def parameters_schema(self) -> dict[str, Any]:
        action_enum = ["search", "fetch"]
        action_description = "Action to perform. Default: auto-detected from query/url."

        if self._use_selenium:
            action_enum = [
                "search",
                "fetch",
                "click",
                "fill",
                "extract_links",
                "execute_js",
                "crawl",
                "get_content",
            ]
            action_description = (
                "Browser action to perform. Default: auto-detected from query/url. "
                "Use 'click' to click elements, 'fill' to fill forms, "
                "'extract_links' to get all links, 'execute_js' to run JavaScript, "
                "'crawl' to recursively browse a site, 'get_content' to read current page."
            )

        properties: dict[str, Any] = {
            "query": {
                "type": "string",
                "description": "Search query. Returns search results with titles, URLs, and snippets.",
            },
            "url": {
                "type": "string",
                "description": "URL of a specific web page to read/open.",
            },
            "fetch_content": {
                "type": "boolean",
                "description": (
                    "If true, automatically fetch and include full content of found pages. Default: false (faster)."
                ),
            },
            "max_results": {
                "type": "integer",
                "description": f"Maximum number of search results. Default: {self._max_results}",
            },
            "action": {
                "type": "string",
                "enum": action_enum,
                "description": action_description,
            },
        }

        if self._use_selenium:
            properties.update(
                {
                    "selector": {
                        "type": "string",
                        "description": (
                            "CSS selector for click/fill actions. "
                            "Examples: 'button.submit', 'input[name=q]', '#login-btn', 'a.nav-link'."
                        ),
                    },
                    "value": {
                        "type": "string",
                        "description": "Value to type into an input field (for action='fill').",
                    },
                    "submit": {
                        "type": "boolean",
                        "description": "Press Enter after filling input (for action='fill'). Default: false.",
                    },
                    "js_code": {
                        "type": "string",
                        "description": (
                            "JavaScript code to execute on the page (for action='execute_js'). "
                            "Use 'return ...' to get a value back."
                        ),
                    },
                    "wait_for_selector": {
                        "type": "string",
                        "description": (
                            "CSS selector to wait for before extracting content. "
                            "Useful for SPA pages. Example: '.main-content', '#article-body'."
                        ),
                    },
                    "max_depth": {
                        "type": "integer",
                        "description": "Maximum crawl depth (for action='crawl'). Default: 2.",
                    },
                    "max_pages": {
                        "type": "integer",
                        "description": "Maximum pages to crawl (for action='crawl'). Default: 10.",
                    },
                    "url_filter": {
                        "type": "string",
                        "description": (
                            "URL prefix filter for crawl/extract_links. "
                            "Only links starting with this prefix are followed. "
                            "Default: same domain as start URL."
                        ),
                    },
                }
            )

        return {
            "type": "object",
            "properties": properties,
            "required": [],
        }

    def _format_search_results(
        self,
        results: list[dict[str, str]],
        *,
        with_content: bool = False,
    ) -> str:
        """Форматировать результаты поиска для вывода."""
        if not results:
            return "No results found for the query."

        lines = [f"Found {len(results)} result(s):\n"]

        for i, result in enumerate(results, 1):
            title = result.get("title", "Untitled")
            url = result.get("url", "")
            snippet = result.get("snippet", "")
            content = result.get("content", "")

            lines.append(f"[{i}] {title}")
            if url:
                lines.append(f"    URL: {url}")

            if content and with_content:
                truncated = content[: self._max_content_length]
                if len(content) > self._max_content_length:
                    truncated += "\n    ... (content truncated)"
                lines.append(f"\n    --- Page Content ---\n    {truncated}\n")
            elif snippet:
                lines.append(f"    {snippet}")

            lines.append("")

        return "\n".join(lines).strip()

    def _get_active_fetcher(self) -> URLFetcher | SeleniumFetcher:
        """Получить активный fetcher (Selenium или стандартный)."""
        if self._use_selenium and self._selenium_fetcher is not None:
            return self._selenium_fetcher
        return self._fetcher

    def _fetch_url(
        self,
        url: str,
        wait_for_selector: str | None = None,
    ) -> ToolResult:
        """Скачать и вернуть содержимое URL."""
        import time as _time

        start = _time.monotonic()
        self._emit_tool_start("fetch", {"url": url, "wait_for_selector": wait_for_selector})

        fetcher = self._get_active_fetcher()

        try:
            if isinstance(fetcher, SeleniumFetcher) and wait_for_selector:
                result = fetcher.fetch_with_wait(url, wait_for_selector=wait_for_selector)
            else:
                result = fetcher.fetch(url)

            elapsed_ms = (_time.monotonic() - start) * 1000

            if not result["success"]:
                self._emit_tool_end(
                    "fetch",
                    success=False,
                    duration_ms=elapsed_ms,
                    result_summary=f"Failed: {result['error']}",
                )
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Failed to fetch URL: {result['error']}",
                )

            output_lines = []
            if result["title"]:
                output_lines.append(f"Title: {result['title']}")
            output_lines.append(f"URL: {url}")
            if self._use_selenium:
                output_lines.append("(Rendered with Selenium browser)")
            output_lines.append("")
            output_lines.append("--- Page Content ---")
            output_lines.append(result["content"])
            output = "\n".join(output_lines)

            self._emit_tool_end(
                "fetch",
                success=True,
                output_size=len(output),
                duration_ms=elapsed_ms,
                result_summary=f"Fetched {url} ({len(result['content'])} chars)",
            )

            return ToolResult(tool_name=self.name, success=True, output=output)

        except Exception as e:  # noqa: BLE001
            self._emit_tool_error("fetch", e)
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    def _fetch_page_content(self, page_url: str) -> dict[str, Any] | None:
        """Скачать содержимое одной страницы через активный fetcher."""
        fetcher = self._get_active_fetcher()
        fetched = fetcher.fetch(page_url)
        if fetched["success"]:
            return fetched
        return None

    # ================================================================
    # Browser Actions (требуют Selenium)
    # ================================================================

    def _require_selenium(self, action: str) -> SeleniumFetcher:
        """Проверить что Selenium доступен, иначе — ошибка."""
        if self._selenium_fetcher is None:
            msg = (
                f"Action '{action}' requires Selenium. "
                "Initialize WebSearchTool with use_selenium=True or provide selenium_fetcher."
            )
            raise RuntimeError(msg)
        return self._selenium_fetcher

    def _execute_click(self, selector: str, wait_timeout: int | None = None) -> ToolResult:
        """Кликнуть по элементу."""
        import time as _time

        start = _time.monotonic()
        self._emit_tool_start("click", {"selector": selector})

        try:
            fetcher = self._require_selenium("click")
            result = fetcher.click_element(selector, wait_timeout=wait_timeout)
            elapsed_ms = (_time.monotonic() - start) * 1000

            if not result["success"]:
                self._emit_tool_end(
                    "click",
                    success=False,
                    duration_ms=elapsed_ms,
                    result_summary=f"Click failed: {result['error']}",
                )
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Click failed: {result['error']}",
                )

            output_parts = [
                f"Clicked element: '{selector}'",
                f"Element text: {result['clicked_text'][:200]}" if result["clicked_text"] else "",
                f"Current URL: {result['url']}",
                f"Page title: {result['title']}",
            ]
            output = "\n".join(p for p in output_parts if p)

            self._emit_tool_end(
                "click",
                success=True,
                output_size=len(output),
                duration_ms=elapsed_ms,
                result_summary=f"Clicked '{selector}' -> {result['url']}",
            )
            return ToolResult(tool_name=self.name, success=True, output=output)

        except Exception as e:  # noqa: BLE001
            self._emit_tool_error("click", e)
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    def _execute_fill(
        self,
        selector: str,
        value: str,
        *,
        submit: bool = False,
        wait_timeout: int | None = None,
    ) -> ToolResult:
        """Заполнить поле ввода."""
        import time as _time

        start = _time.monotonic()
        self._emit_tool_start("fill", {"selector": selector, "value": value, "submit": submit})

        try:
            fetcher = self._require_selenium("fill")
            result = fetcher.fill_input(
                selector,
                value,
                submit=submit,
                wait_timeout=wait_timeout,
            )
            elapsed_ms = (_time.monotonic() - start) * 1000

            if not result["success"]:
                self._emit_tool_end(
                    "fill",
                    success=False,
                    duration_ms=elapsed_ms,
                    result_summary=f"Fill failed: {result['error']}",
                )
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Fill failed: {result['error']}",
                )

            output_parts = [
                f"Filled '{selector}' with value: '{value}'",
                f"Submitted: {submit}",
                f"Current URL: {result['url']}",
                f"Page title: {result['title']}",
            ]
            output = "\n".join(output_parts)

            self._emit_tool_end(
                "fill",
                success=True,
                output_size=len(output),
                duration_ms=elapsed_ms,
                result_summary=f"Filled '{selector}' with '{value[:50]}'",
            )
            return ToolResult(tool_name=self.name, success=True, output=output)

        except Exception as e:  # noqa: BLE001
            self._emit_tool_error("fill", e)
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    def _execute_extract_links(
        self,
        url: str | None = None,
        *,
        selector: str = "a[href]",
        url_filter: str | None = None,
        max_links: int = 50,
    ) -> ToolResult:
        """Извлечь ссылки со страницы."""
        import time as _time

        start = _time.monotonic()
        self._emit_tool_start("extract_links", {"url": url, "selector": selector})

        try:
            fetcher = self._require_selenium("extract_links")

            # Если указан URL — сначала открываем страницу
            if url:
                fetch_result = fetcher.fetch(url)
                if not fetch_result["success"]:
                    self._emit_tool_end(
                        "extract_links",
                        success=False,
                        result_summary=f"Failed to open {url}",
                    )
                    return ToolResult(
                        tool_name=self.name,
                        success=False,
                        error=f"Failed to open URL: {fetch_result['error']}",
                    )

            result = fetcher.extract_links(
                selector=selector,
                base_url_filter=url_filter,
                max_links=max_links,
            )
            elapsed_ms = (_time.monotonic() - start) * 1000

            if not result["success"]:
                self._emit_tool_end(
                    "extract_links",
                    success=False,
                    duration_ms=elapsed_ms,
                    result_summary=f"Extract failed: {result['error']}",
                )
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Extract links failed: {result['error']}",
                )

            lines = [f"Found {result['count']} link(s) on {result['url']}:\n"]
            for i, link in enumerate(result["links"], 1):
                text = link.get("text", "").strip() or "(no text)"
                lines.append(f"[{i}] {text}")
                lines.append(f"    URL: {link['url']}")
                if link.get("title"):
                    lines.append(f"    Title: {link['title']}")
                lines.append("")

            output = "\n".join(lines).strip()

            self._emit_tool_end(
                "extract_links",
                success=True,
                output_size=len(output),
                duration_ms=elapsed_ms,
                result_summary=f"Extracted {result['count']} links from {result['url']}",
            )
            return ToolResult(tool_name=self.name, success=True, output=output)

        except Exception as e:  # noqa: BLE001
            self._emit_tool_error("extract_links", e)
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    def _execute_js(self, js_code: str) -> ToolResult:
        """Выполнить JavaScript на текущей странице."""
        import time as _time

        start = _time.monotonic()
        self._emit_tool_start("execute_js", {"js_code": js_code[:200]})

        try:
            fetcher = self._require_selenium("execute_js")
            result = fetcher.execute_js(js_code)
            elapsed_ms = (_time.monotonic() - start) * 1000

            if not result["success"]:
                self._emit_tool_end(
                    "execute_js",
                    success=False,
                    duration_ms=elapsed_ms,
                    result_summary=f"JS error: {result['error']}",
                )
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"JavaScript execution failed: {result['error']}",
                )

            output_parts = [
                f"JavaScript executed on: {result['url']}",
            ]
            if result["return_value"] is not None:
                output_parts.append(f"Return value: {result['return_value']}")
            else:
                output_parts.append("(no return value)")

            output = "\n".join(output_parts)

            self._emit_tool_end(
                "execute_js",
                success=True,
                output_size=len(output),
                duration_ms=elapsed_ms,
                result_summary="JS executed successfully",
            )
            return ToolResult(tool_name=self.name, success=True, output=output)

        except Exception as e:  # noqa: BLE001
            self._emit_tool_error("execute_js", e)
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    def _execute_crawl(
        self,
        start_url: str,
        *,
        max_pages: int = 10,
        max_depth: int = 2,
        url_filter: str | None = None,
    ) -> ToolResult:
        """Рекурсивно обойти сайт."""
        import time as _time

        start = _time.monotonic()
        self._emit_tool_start(
            "crawl",
            {
                "url": start_url,
                "max_pages": max_pages,
                "max_depth": max_depth,
                "url_filter": url_filter,
            },
        )

        try:
            fetcher = self._require_selenium("crawl")
            result = fetcher.crawl(
                start_url,
                max_pages=max_pages,
                max_depth=max_depth,
                url_filter=url_filter,
                extract_content=True,
            )
            elapsed_ms = (_time.monotonic() - start) * 1000

            if not result["success"] and not result["pages"]:
                self._emit_tool_end(
                    "crawl",
                    success=False,
                    duration_ms=elapsed_ms,
                    result_summary=f"Crawl failed: {result['error']}",
                )
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Crawl failed: {result['error']}",
                )

            lines = [f"Crawled {result['total_pages']} page(s) starting from {start_url}:\n"]
            for i, page in enumerate(result["pages"], 1):
                lines.append(f"[{i}] {page.get('title', 'Untitled')}")
                lines.append(f"    URL: {page['url']}")
                lines.append(f"    Depth: {page['depth']}, Links found: {page.get('links_found', 0)}")
                content = page.get("content", "")
                if content:
                    truncated = content[: self._max_content_length]
                    if len(content) > self._max_content_length:
                        truncated += "\n    ... (content truncated)"
                    lines.append(f"\n    --- Page Content ---\n    {truncated}\n")
                lines.append("")

            if result.get("error"):
                lines.append(f"Note: Crawl completed with warning: {result['error']}")

            output = "\n".join(lines).strip()

            self._emit_tool_end(
                "crawl",
                success=True,
                output_size=len(output),
                duration_ms=elapsed_ms,
                result_summary=f"Crawled {result['total_pages']} pages from {start_url}",
            )
            return ToolResult(tool_name=self.name, success=True, output=output)

        except Exception as e:  # noqa: BLE001
            self._emit_tool_error("crawl", e)
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    def _execute_get_content(self) -> ToolResult:
        """Получить содержимое текущей открытой страницы."""
        import time as _time

        start = _time.monotonic()
        self._emit_tool_start("get_content", {})

        try:
            fetcher = self._require_selenium("get_content")
            result = fetcher.get_page_content()
            elapsed_ms = (_time.monotonic() - start) * 1000

            if not result["success"]:
                self._emit_tool_end(
                    "get_content",
                    success=False,
                    duration_ms=elapsed_ms,
                    result_summary=f"Failed: {result['error']}",
                )
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error=f"Get content failed: {result['error']}",
                )

            output_lines = []
            if result["title"]:
                output_lines.append(f"Title: {result['title']}")
            output_lines.append(f"URL: {result['url']}")
            output_lines.append("")
            output_lines.append("--- Page Content ---")
            output_lines.append(result["content"])
            output = "\n".join(output_lines)

            self._emit_tool_end(
                "get_content",
                success=True,
                output_size=len(output),
                duration_ms=elapsed_ms,
                result_summary=f"Got content from {result['url']} ({len(result['content'])} chars)",
            )
            return ToolResult(tool_name=self.name, success=True, output=output)

        except Exception as e:  # noqa: BLE001
            self._emit_tool_error("get_content", e)
            return ToolResult(tool_name=self.name, success=False, error=str(e))

    # ================================================================
    # Main execute
    # ================================================================

    def execute(  # noqa: PLR0912
        self,
        query: str = "",
        url: str = "",
        *,
        action: str = "",
        fetch_content: bool | None = None,
        max_results: int | None = None,
        wait_for_selector: str | None = None,
        selector: str = "",
        value: str = "",
        submit: bool = False,
        js_code: str = "",
        max_depth: int = 2,
        max_pages: int = 10,
        url_filter: str | None = None,
        **_kwargs: Any,
    ) -> ToolResult:
        """
        Выполнить поиск, прочитать веб-страницу или выполнить browser action.

        Args:
            query: Поисковый запрос.
            url: URL для чтения/открытия страницы.
            action: Действие: "search", "fetch", "click", "fill",
                   "extract_links", "execute_js", "crawl", "get_content".
                   Если не указано — определяется автоматически.
            fetch_content: Скачивать ли содержимое страниц.
            max_results: Максимальное количество результатов.
            wait_for_selector: CSS-селектор для ожидания.
            selector: CSS-селектор для click/fill.
            value: Значение для fill.
            submit: Нажать Enter после fill.
            js_code: JavaScript код для execute_js.
            max_depth: Максимальная глубина для crawl.
            max_pages: Максимальное количество страниц для crawl.
            url_filter: Фильтр URL для crawl/extract_links.

        Returns:
            ToolResult с результатами.

        """
        # Определяем action автоматически если не указан
        if not action:
            if query:
                action = "search"
            elif url:
                action = "fetch"
            elif selector:
                action = "click"
            elif js_code:
                action = "execute_js"
            else:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error="No action, query, url, selector, or js_code provided.",
                )

        # Dispatch по action
        if action == "click":
            if not selector:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error="Action 'click' requires 'selector' parameter.",
                )
            return self._execute_click(selector)

        if action == "fill":
            if not selector:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error="Action 'fill' requires 'selector' parameter.",
                )
            return self._execute_fill(selector, value, submit=submit)

        if action == "extract_links":
            return self._execute_extract_links(
                url or None,
                selector=selector or "a[href]",
                url_filter=url_filter,
            )

        if action == "execute_js":
            if not js_code:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error="Action 'execute_js' requires 'js_code' parameter.",
                )
            return self._execute_js(js_code)

        if action == "crawl":
            if not url:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error="Action 'crawl' requires 'url' parameter.",
                )
            return self._execute_crawl(
                url,
                max_pages=max_pages,
                max_depth=max_depth,
                url_filter=url_filter,
            )

        if action == "get_content":
            return self._execute_get_content()

        if action == "fetch":
            if not url:
                return ToolResult(
                    tool_name=self.name,
                    success=False,
                    error="Action 'fetch' requires 'url' parameter.",
                )
            return self._fetch_url(url, wait_for_selector=wait_for_selector)

        # action == "search" или по умолчанию
        if not query:
            return ToolResult(
                tool_name=self.name,
                success=False,
                error="No search query provided. Use 'query' to search or 'url' to read a page.",
            )

        return self._execute_search(
            query,
            fetch_content=fetch_content,
            max_results=max_results,
            wait_for_selector=wait_for_selector,
        )

    def _execute_search(
        self,
        query: str,
        *,
        fetch_content: bool | None = None,
        max_results: int | None = None,
        wait_for_selector: str | None = None,  # noqa: ARG002
    ) -> ToolResult:
        """Выполнить поиск."""
        import time as _time

        start = _time.monotonic()
        num_results = max_results if max_results is not None else self._max_results
        num_results = max(1, min(num_results, 10))
        should_fetch = fetch_content if fetch_content is not None else self._fetch_content

        self._emit_tool_start("search", {"query": query, "max_results": num_results, "fetch_content": should_fetch})

        try:
            results = self._provider.search(query, num_results)

            if not results:
                output = (
                    f"No results found for query: '{query}'. Try rephrasing your search or using different keywords."
                )
                elapsed_ms = (_time.monotonic() - start) * 1000
                self._emit_tool_end(
                    "search",
                    success=True,
                    output_size=len(output),
                    duration_ms=elapsed_ms,
                    result_summary="No results found",
                )
                return ToolResult(tool_name=self.name, success=True, output=output)

            # Если нужно скачать содержимое страниц
            if should_fetch:
                for result in results:
                    page_url = result.get("url", "")
                    if page_url:
                        fetched = self._fetch_page_content(page_url)
                        if fetched is not None:
                            result["content"] = fetched["content"]
                            if fetched["title"] and not result.get("title"):
                                result["title"] = fetched["title"]

            output = self._format_search_results(results, with_content=should_fetch)
            elapsed_ms = (_time.monotonic() - start) * 1000

            self._emit_tool_end(
                "search",
                success=True,
                output_size=len(output),
                duration_ms=elapsed_ms,
                result_summary=f"Found {len(results)} results for '{query}'",
            )

            return ToolResult(tool_name=self.name, success=True, output=output)

        except TimeoutError as e:
            self._emit_tool_error("search", e)
            error_msg = f"Search timed out after {self._timeout} seconds"
        except urllib.error.URLError as e:
            self._emit_tool_error("search", e)
            error_msg = f"Network error: {e.reason}"
        except (ValueError, KeyError, OSError) as e:
            self._emit_tool_error("search", e)
            error_msg = f"Search error: {e}"

        return ToolResult(tool_name=self.name, success=False, error=error_msg)

    def close(self) -> None:
        """Закрыть Selenium браузер (если используется)."""
        if self._selenium_fetcher is not None:
            self._selenium_fetcher.close()

    def __del__(self) -> None:
        """Закрыть ресурсы при удалении."""
        self.close()

    def __enter__(self) -> Self:
        """Поддержка context manager."""
        return self

    def __exit__(self, *_args: object) -> None:
        """Закрыть ресурсы при выходе из context manager."""
        self.close()


# ============================================================
# Фабрика для создания WebSearchTool из dict-конфига
# ============================================================


def _create_web_search_tool(**kwargs: Any) -> WebSearchTool:
    """
    Создать WebSearchTool из параметров конфига.

    Поддерживает все параметры конструктора WebSearchTool.
    Дополнительно: ``provider="serper"`` / ``provider="tavily"``
    автоматически создаёт соответствующий провайдер.

    Example конфиг::

        {"name": "web_search", "use_selenium": True}
        {"name": "web_search", "provider": "serper", "fetch_content": True}

    """
    provider = kwargs.pop("provider", None)
    if isinstance(provider, str):
        provider_name = provider.lower()
        if provider_name == "serper":
            api_key = kwargs.pop("serper_api_key", None) or kwargs.pop("api_key", None)
            provider = SerperProvider(api_key=api_key) if api_key else DuckDuckGoProvider()
        elif provider_name == "tavily":
            api_key = kwargs.pop("tavily_api_key", None) or kwargs.pop("api_key", None)
            provider = TavilyProvider(api_key=api_key) if api_key else DuckDuckGoProvider()
        elif provider_name in ("duckduckgo", "ddg"):
            provider = DuckDuckGoProvider()
        else:
            provider = DuckDuckGoProvider()

    return WebSearchTool(provider=provider, **kwargs)


# Автоматическая регистрация фабрики при импорте модуля
from .base import register_tool_factory  # noqa: E402

register_tool_factory("web_search", _create_web_search_tool)
