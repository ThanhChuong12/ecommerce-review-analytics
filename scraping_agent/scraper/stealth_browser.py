"""
stealth_browser.py — CloakBrowser-powered browser factory for all scrapers.

Priority:
  1. CloakBrowser 0.3.28 (installed) -> patched Chromium binary
     Binary: ~/.cloakbrowser/chromium-146.x/chrome.exe
     Bypasses: Cloudflare, DataDome, FingerprintJS, reCAPTCHA v3 (score 0.9)
  2. Playwright fallback (if CloakBrowser unavailable)

Public API (used by shopee.py, lazada.py, generic_playwright.py):
    context = await launch_stealth_context(
        storage_state="output/agent_sessions/state_shopee.json",
        headless=True,
        humanize=False,
        locale="vi-VN",
        timezone="Asia/Ho_Chi_Minh",
    )
    page = await context.new_page()
    ...
    await context.close()  # also closes browser + Playwright instance
"""

from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from typing import Any

# Disable CloakBrowser auto-update check — prevents blocking network call on startup
os.environ.setdefault("CLOAKBROWSER_NO_UPDATE_CHECK", "1")

log = logging.getLogger("StealthBrowser")

# ── Detect CloakBrowser ──────────────────────────────────────────────────────
try:
    import cloakbrowser as _cloak
    _CLOAK_VERSION = _cloak.__version__
    _CLOAK_AVAILABLE = True
    log.info("CloakBrowser %s detected — stealth Chromium active", _CLOAK_VERSION)
except ImportError:
    _CLOAK_AVAILABLE = False
    _CLOAK_VERSION = None
    log.info("CloakBrowser not installed — using Playwright fallback")

# ── Playwright fallback stealth args ────────────────────────────────────────
_PLAYWRIGHT_ARGS = [
    "--disable-blink-features=AutomationControlled",
    "--no-sandbox",
    "--disable-dev-shm-usage",
    "--disable-gpu",
]

_DEFAULT_UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)


# ── Main public function ─────────────────────────────────────────────────────

async def launch_stealth_context(
    storage_state: str | None = None,
    headless: bool = True,
    humanize: bool = False,
    human_preset: str = "default",   # 'default' | 'careful'
    locale: str = "vi-VN",
    timezone: str = "Asia/Ho_Chi_Minh",
    proxy: str | None = None,
    user_data_dir: str | None = None,
    **extra_ctx_kwargs: Any,
) -> Any:
    """
    Launch a stealth browser context. Returns a Playwright BrowserContext.

    Args:
        storage_state: Path to saved session JSON (cookies/localStorage).
                       Loaded if file exists, ignored if not.
        headless: Run without visible browser window.
        humanize: Enable human-like mouse/keyboard behavior (CloakBrowser only).
        locale: Browser locale string e.g. "vi-VN".
        timezone: IANA timezone e.g. "Asia/Ho_Chi_Minh".
        proxy: Proxy URL e.g. "http://user:pass@host:port" or "socks5://host:port".
        user_data_dir: Persistent profile folder path (keeps cookies across runs).
                       If set, uses launch_persistent_context for better session.
        **extra_ctx_kwargs: Passed directly to browser.new_context().

    Returns:
        Playwright BrowserContext. Call await context.close() when done.
    """
    # Validate storage_state path
    if storage_state and not Path(storage_state).exists():
        log.debug("storage_state not found, starting fresh: %s", storage_state)
        storage_state = None

    if _CLOAK_AVAILABLE:
        return await _cloak_launch_context(
            storage_state=storage_state,
            headless=headless,
            humanize=humanize,
            human_preset=human_preset,
            locale=locale,
            timezone=timezone,
            proxy=proxy,
            user_data_dir=user_data_dir,
            **extra_ctx_kwargs,
        )
    return await _playwright_launch_context(
        storage_state=storage_state,
        headless=headless,
        locale=locale,
        timezone=timezone,
        proxy=proxy,
        **extra_ctx_kwargs,
    )


def is_cloak_available() -> bool:
    """Return True if CloakBrowser is installed and binary is ready."""
    return _CLOAK_AVAILABLE


def cloak_info() -> dict:
    """Return CloakBrowser installation info."""
    if not _CLOAK_AVAILABLE:
        return {"installed": False, "version": None}
    try:
        info = _cloak.binary_info()
        info["version"] = _CLOAK_VERSION
        return info
    except Exception:
        return {"installed": True, "version": _CLOAK_VERSION}


# ── CloakBrowser implementation ───────────────────────────────────────────────

async def _cloak_launch_context(
    storage_state: str | None,
    headless: bool,
    humanize: bool,
    human_preset: str,
    locale: str,
    timezone: str,
    proxy: str | None,
    user_data_dir: str | None,
    **extra_ctx_kwargs: Any,
) -> Any:
    """Launch using CloakBrowser's patched Chromium binary."""
    from cloakbrowser import launch_context_async, launch_persistent_context_async

    common_kwargs: dict[str, Any] = {
        "headless": headless,
        "humanize": humanize,
        "locale": locale,
        "timezone": timezone,
    }
    if humanize:
        common_kwargs["human_preset"] = human_preset
    if proxy:
        common_kwargs["proxy"] = proxy

    # Persistent profile: better for session persistence + avoids incognito detection
    if user_data_dir:
        log.info(
            "[CloakBrowser] persistent context headless=%s humanize=%s profile=%s",
            headless, humanize, user_data_dir,
        )
        ctx = await launch_persistent_context_async(
            user_data_dir=user_data_dir,
            **common_kwargs,
            **extra_ctx_kwargs,
        )
        return ctx

    # Ephemeral context with optional storage_state
    if storage_state:
        extra_ctx_kwargs["storage_state"] = storage_state

    log.info(
        "[CloakBrowser] context headless=%s humanize=%s storage=%s",
        headless, humanize, bool(storage_state),
    )
    ctx = await launch_context_async(
        **common_kwargs,
        **extra_ctx_kwargs,
    )
    return ctx


# ── Playwright fallback implementation ────────────────────────────────────────

async def _playwright_launch_context(
    storage_state: str | None,
    headless: bool,
    locale: str,
    timezone: str,
    proxy: str | None,
    **extra_ctx_kwargs: Any,
) -> Any:
    """Fallback: plain Playwright with stealth args."""
    from playwright.async_api import async_playwright

    log.info("[Playwright] fallback context headless=%s storage=%s", headless, bool(storage_state))

    pw = await async_playwright().start()

    launch_kwargs: dict[str, Any] = {
        "headless": headless,
        "args": _PLAYWRIGHT_ARGS,
    }
    if proxy:
        launch_kwargs["proxy"] = {"server": proxy}

    browser = await pw.chromium.launch(**launch_kwargs)

    ctx_kwargs: dict[str, Any] = {
        "user_agent": _DEFAULT_UA,
        "locale": locale,
        "timezone_id": timezone,
        "viewport": {"width": 1440, "height": 900},
        "extra_http_headers": {"Accept-Language": f"{locale},vi;q=0.9,en-US;q=0.8"},
        **extra_ctx_kwargs,
    }
    if storage_state:
        ctx_kwargs["storage_state"] = storage_state

    context = await browser.new_context(**ctx_kwargs)

    # Patch navigator.webdriver
    await context.add_init_script(
        "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
    )

    # Keep references alive + cleanup on close
    _orig_close = context.close

    async def _close_all() -> None:
        try:
            await _orig_close()
        finally:
            try:
                await browser.close()
            finally:
                await pw.stop()

    context.close = _close_all
    return context


# ── Retry decorator ───────────────────────────────────────────────────────────

def with_retry(max_retries: int = 3, base_delay: float = 2.0):
    """
    Decorator: retry async function with exponential backoff.

    Usage:
        @with_retry(max_retries=3, base_delay=2.0)
        async def fetch(url):
            ...
    """
    import functools

    def decorator(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kwargs):
            last_exc: Exception | None = None
            for attempt in range(max_retries):
                try:
                    return await fn(*args, **kwargs)
                except Exception as exc:
                    last_exc = exc
                    wait = base_delay * (2 ** attempt)
                    log.warning(
                        "[Retry] %s attempt %d/%d failed: %s — retry in %.1fs",
                        fn.__name__, attempt + 1, max_retries, exc, wait,
                    )
                    await asyncio.sleep(wait)
            raise last_exc  # type: ignore[misc]
        return wrapper
    return decorator
