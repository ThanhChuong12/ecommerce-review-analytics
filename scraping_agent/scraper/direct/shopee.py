"""
shopee.py — ShopeeScraper: Playwright network interception cho Shopee reviews.

Cơ chế: Giống LazadaScraper
  1. page.on("response") → bắt GET api/v4/product/get_ratings
  2. Navigate + scroll → trang 1 tự động intercepted
  3. Click nút phân trang lặp lại cho đến khi đủ hoặc hết

URL pattern:  https://shopee.vn/{slug}-i.{shopId}.{itemId}
API:          GET https://shopee.vn/api/v4/product/get_ratings
              ?itemid=X&shopid=Y&offset=0&limit=6&type=0
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from scraper.exporter import ReviewExporter
from scraper.models import Review

log = logging.getLogger("ShopeeScraper")

SESSION_DIR    = Path("output") / "agent_sessions"
_RATINGS_PATH  = "api/v4/product/get_ratings"
PAGE_LIMIT     = 6        # Shopee hiển thị 6 review/trang
CAPTCHA_AUTO   = 90.0     # giây chờ tự động trước khi hỏi user
CAPTCHA_CHUNK  = 15.0

# Selectors nút trang tiếp theo của Shopee
_NEXT_SELECTORS = [
    "button.shopee-icon-button--right:not([disabled])",
    ".shopee-page-controller button:last-child:not([disabled])",
    "[class*='page-controller'] button:last-child:not([disabled])",
    "button[aria-label='next']:not([disabled])",
    "button[aria-label='Next']:not([disabled])",
]

# Selectors vùng review để scroll đến
_REVIEW_SECTION = [
    ".shopee-product-rating",
    "[class*='product-rating']",
    ".page-product__reviews",
    "[class*='review']",
    ".shopee-page-controller",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_ids(url: str) -> tuple[str, str]:
    """Trả về (shop_id, item_id) từ Shopee URL.

    Patterns:
      shopee.vn/name-i.{shopId}.{itemId}
      shopee.vn/{shopId}/name-{itemId}
    """
    path = urlparse(url).path.rstrip("/")
    m = re.search(r"-i\.(\d+)\.(\d+)$", path)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"[.-]i[.-](\d+)[.-](\d+)", path)
    if m:
        return m.group(1), m.group(2)
    raise ValueError(
        f"Không tìm thấy shopId/itemId trong URL: {url}\n"
        "URL Shopee chuẩn: shopee.vn/product-name-i.{{shopId}}.{{itemId}}"
    )


def _extract_payload(data: dict) -> tuple[list[dict], int]:
    """Trích (ratings_list, total) từ Shopee get_ratings response."""
    inner   = data.get("data") or {}
    ratings = inner.get("ratings") or []
    summary = inner.get("item_rating_summary") or {}
    total   = int(summary.get("rating_total") or 0)
    return ratings, total


def _parse_date(ctime) -> str:
    try:
        ts = int(ctime)
        if ts > 1e12:
            ts //= 1000
        return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return str(ctime) if ctime else ""


def _normalize(raw: dict, product_url: str, product_name: str) -> Review | None:
    try:
        review_id  = str(raw.get("rating_id") or raw.get("id") or "")
        rating     = int(raw.get("rating_star") or 0)
        text       = " ".join(str(raw.get("comment") or "").split())
        date_str   = _parse_date(raw.get("ctime"))
        image_urls = [img for img in (raw.get("images") or []) if img]
        return Review(
            review_id=review_id,
            product_name=product_name,
            text=text,
            rating=rating,
            date=date_str,
            image_urls=image_urls,
            product_url=product_url,
        )
    except Exception as exc:
        log.debug("Skip malformed review: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class ShopeeScraper:
    """Shopee review scraper dùng Playwright network interception."""

    SITE_NAME = "Shopee"

    def __init__(
        self,
        headless: bool = False,
        max_pages: int = 100,
        delay: float   = 1.5,
    ) -> None:
        self.headless   = headless
        self.max_pages  = max_pages
        self.delay      = delay
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        self._state_file = SESSION_DIR / "state_shopee.vn.json"
        self._seen: set[str] = set()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        url: str,
        output_path: str,
        fmt: str         = "csv",
        max_reviews: int = 3000,
    ) -> int:
        print(f"  [{self.SITE_NAME}] Playwright network interception")
        print(f"  headless={self.headless} | max_reviews={max_reviews:,}")

        try:
            shop_id, item_id = _parse_ids(url)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        print(f"  shopId={shop_id} | itemId={item_id}")

        exporter = ReviewExporter(output_path, fmt)
        raw_reviews, product_name = await self._scrape_async(url, max_reviews)

        batch: list[Review] = []
        for raw in raw_reviews:
            r = _normalize(raw, url, product_name)
            if r:
                batch.append(r)

        total_saved = exporter.save_batch(batch) if batch else 0
        print(f"  [{self.SITE_NAME}] Tổng: {total_saved:,} reviews")
        return total_saved

    # ------------------------------------------------------------------
    # Playwright core
    # ------------------------------------------------------------------

    async def _scrape_async(
        self, product_url: str, max_reviews: int
    ) -> tuple[list[dict], str]:
        from playwright.async_api import async_playwright

        all_raw: list[dict]  = []
        total_on_server: int = 0
        page_num: int        = 1
        product_name: str    = ""
        _seen_raw: set[str]  = set()
        got_response         = asyncio.Event()

        async def _on_response(response) -> None:
            nonlocal total_on_server
            if _RATINGS_PATH not in response.url or response.status != 200:
                return
            try:
                body   = await response.text()
                data   = json.loads(body)
                items, total = _extract_payload(data)

                new_items: list[dict] = []
                for item in items:
                    rid = str(item.get("rating_id") or item.get("id") or "")
                    key = f"id:{rid}" if rid else str(item)[:80]
                    if key not in _seen_raw:
                        _seen_raw.add(key)
                        new_items.append(item)

                all_raw.extend(new_items)
                if total > total_on_server:
                    total_on_server = total
                if new_items:
                    got_response.set()
                    log.info("[Shopee] +%d reviews (total=%d)", len(new_items), len(all_raw))
            except Exception as exc:
                log.debug("[Shopee] Parse error: %s", exc)

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
                    headless=self.headless,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                    ],
                )

                ctx_kwargs: dict = {
                    "user_agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "locale": "vi-VN",
                    "timezone_id": "Asia/Ho_Chi_Minh",
                    "viewport": {"width": 1440, "height": 900},
                    "extra_http_headers": {"Accept-Language": "vi-VN,vi;q=0.9"},
                }
                if self._state_file.exists():
                    ctx_kwargs["storage_state"] = str(self._state_file)

                context = await browser.new_context(**ctx_kwargs)
                await context.add_init_script(
                    "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
                )
                page = await context.new_page()
                page.on("response", _on_response)

                # ── Bước 1: Load trang, chờ review API ─────────────────────
                got_response.clear()
                await page.goto(product_url, wait_until="domcontentloaded", timeout=35_000)
                await page.wait_for_timeout(2000)

                # Lấy tên sản phẩm
                try:
                    h1 = page.locator("h1").first
                    if await h1.count() > 0:
                        product_name = (await h1.inner_text()).strip()
                    if not product_name:
                        title = await page.title()
                        product_name = title.split("|")[0].strip()
                except Exception:
                    pass

                # Scroll để trigger review section
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.7)")
                await page.wait_for_timeout(1500)

                # Scroll đến khu vực review
                for sel in _REVIEW_SECTION:
                    try:
                        el = page.locator(sel).first
                        if await el.count() > 0:
                            await el.scroll_into_view_if_needed(timeout=2000)
                            await page.wait_for_timeout(800)
                            break
                    except Exception:
                        continue

                # Chờ response trang 1
                page1_ok = False
                try:
                    await asyncio.wait_for(_await_event(got_response), timeout=30.0)
                    page1_ok = len(all_raw) > 0
                except asyncio.TimeoutError:
                    pass

                if not page1_ok:
                    # Fallback: scroll full
                    got_response.clear()
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(2000)
                    try:
                        await asyncio.wait_for(_await_event(got_response), timeout=15.0)
                        page1_ok = len(all_raw) > 0
                    except asyncio.TimeoutError:
                        pass

                if not page1_ok:
                    page1_ok = await _captcha_pause_and_resume(
                        got_response,
                        check_fn=lambda: len(all_raw) > 0,
                        context="trang đầu tiên",
                    )

                print(
                    f"  [OK] Trang 1: +{len(all_raw)} reviews"
                    f" | Server total: {total_on_server:,}"
                )

                if not page1_ok:
                    await browser.close()
                    return all_raw, product_name

                # Lưu session
                try:
                    await context.storage_state(path=str(self._state_file))
                except Exception:
                    pass

                # ── Bước 2: Phân trang ─────────────────────────────────────
                while len(all_raw) < max_reviews and page_num < self.max_pages:
                    if total_on_server > 0:
                        total_pages = (total_on_server + PAGE_LIMIT - 1) // PAGE_LIMIT
                        if page_num >= total_pages:
                            print(f"  [Shopee] Đã hết {total_pages} trang.")
                            break

                    clicked = await self._click_next(page, got_response)
                    if not clicked:
                        print("  [Shopee] Không còn trang tiếp theo.")
                        break

                    page_num += 1
                    print(f"  [OK] Trang {page_num}: tổng {len(all_raw):,}/{total_on_server:,}")
                    await page.wait_for_timeout(int(self.delay * 1000))

                await browser.close()

        except Exception as exc:
            log.error("[Shopee] Lỗi: %s", exc, exc_info=True)
            print(f"  [Shopee] Lỗi: {exc}")

        return all_raw, product_name

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    async def _click_next(self, page, got_response: asyncio.Event) -> bool:
        """Click nút trang tiếp theo. Trả về True nếu thành công."""
        # JavaScript click (primary)
        try:
            got_response.clear()
            clicked: bool = await page.evaluate("""
                () => {
                    const sels = [
                        'button.shopee-icon-button--right:not([disabled])',
                        '.shopee-page-controller button:last-child:not([disabled])',
                        'button[aria-label="next"]:not([disabled])',
                    ];
                    for (const s of sels) {
                        const el = document.querySelector(s);
                        if (el && !el.disabled) {
                            el.scrollIntoView({block:'center',behavior:'instant'});
                            el.click();
                            return true;
                        }
                    }
                    return false;
                }
            """)
            if clicked:
                try:
                    await asyncio.wait_for(_await_event(got_response), timeout=10.0)
                    return True
                except asyncio.TimeoutError:
                    return await _captcha_pause_and_resume(
                        got_response, lambda: True, "phân trang"
                    )
        except Exception as exc:
            log.debug("[Shopee] JS click error: %s", exc)

        # Playwright fallback
        for sel in _NEXT_SELECTORS:
            try:
                btn = page.locator(sel).first
                if await btn.count() == 0 or not await btn.is_enabled():
                    continue
                got_response.clear()
                await btn.scroll_into_view_if_needed(timeout=2000)
                await btn.click(force=True, timeout=4000)
                try:
                    await asyncio.wait_for(_await_event(got_response), timeout=10.0)
                    return True
                except asyncio.TimeoutError:
                    pass
            except Exception:
                continue

        return False


# ---------------------------------------------------------------------------
# Utilities (shared với lazada.py)
# ---------------------------------------------------------------------------

async def _await_event(event: asyncio.Event) -> None:
    while not event.is_set():
        await asyncio.sleep(0.05)
    event.clear()


async def _captcha_pause_and_resume(
    got_response: asyncio.Event,
    check_fn,
    context: str = "phân trang",
) -> bool:
    """Chờ tự động → hỏi user nếu vẫn bị block."""
    deadline = CAPTCHA_AUTO
    while deadline > 0:
        got_response.clear()
        chunk = min(CAPTCHA_CHUNK, deadline)
        try:
            await asyncio.wait_for(_await_event(got_response), timeout=chunk)
            if check_fn():
                print("  [Shopee] Tự động nhận được response — tiếp tục.")
                return True
        except asyncio.TimeoutError:
            pass
        deadline -= chunk

    bar = "─" * 60
    while True:
        print(f"\n  {bar}")
        print(f"  [Shopee] BỊ CHẶN BOT ({context})")
        print(f"  {bar}")
        print("  Shopee đã kích hoạt captcha / bot-check.")
        print("  Hãy giải thủ công trong cửa sổ browser.")
        print("  ▸ Nhấn ENTER tiếp tục  |  ▸ Nhập 'n' để dừng")
        print(f"  {bar}")

        loop = asyncio.get_event_loop()
        try:
            ans = await loop.run_in_executor(None, lambda: input("  [Enter/n]: ").strip().lower())
        except (EOFError, KeyboardInterrupt):
            return False

        if ans == "n":
            return False

        deadline2 = CAPTCHA_AUTO
        while deadline2 > 0:
            got_response.clear()
            chunk = min(CAPTCHA_CHUNK, deadline2)
            try:
                await asyncio.wait_for(_await_event(got_response), timeout=chunk)
                if check_fn():
                    print("  [Shopee] Nhận được response sau khi giải captcha!")
                    return True
            except asyncio.TimeoutError:
                pass
            deadline2 -= chunk
