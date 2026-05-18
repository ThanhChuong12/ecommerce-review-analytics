"""
generic_playwright.py — GenericPlaywrightScraper: tự phát hiện review API cho bất kỳ site nào.

Chiến lược:
  1. Navigate đến URL sản phẩm
  2. Intercept TẤT CẢ JSON response trong khi scroll / tương tác
  3. Áp dụng heuristics để tìm list chứa review (có rating + text field)
  4. Normalize về Review model
  5. Thử click nút phân trang phổ biến

Heuristics nhận biết review:
  - Item là dict có field "rating" (int 1-5) VÀ field "text" / "comment" / "content"
  - List có ít nhất 1 item thỏa điều kiện trên
  - Text field không rỗng và dài > 3 ký tự

Dùng khi:
  - Site chưa có scraper riêng
  - Dispatcher không nhận diện được domain
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path

from scraper.exporter import ReviewExporter
from scraper.models import Review

log = logging.getLogger("GenericScraper")

SESSION_DIR   = Path("output") / "agent_sessions"
CAPTCHA_AUTO  = 90.0
CAPTCHA_CHUNK = 15.0

# Tên field điển hình trong các review API
_RATING_KEYS = {
    "rating", "rating_star", "star", "stars", "score",
    "rate", "rate_star", "ratingstar", "overall_rating",
    "overallRating", "reviewScore",
}
_TEXT_KEYS = {
    "comment", "content", "text", "body", "review",
    "review_content", "reviewContent", "message",
    "description", "feedback", "reviewText",
}
_DATE_KEYS = {
    "date", "time", "ctime", "created_at", "createdat",
    "timestamp", "created", "reviewtime", "gmtcreatetime",
    "publishedat", "updated_at",
}
_IMG_KEYS = {
    "image", "images", "image_url", "imageUrl", "img",
    "pictures", "photos", "media",
}

# Generic pagination selectors
_NEXT_SELS = [
    "button[aria-label*='next' i]:not([disabled])",
    "button[aria-label*='tiếp' i]:not([disabled])",
    "a[rel='next']",
    ".pagination .next:not(.disabled)",
    "[class*='pagination'] [class*='next']:not([disabled])",
    "[class*='page'] button:last-child:not([disabled])",
]


# ---------------------------------------------------------------------------
# Heuristics
# ---------------------------------------------------------------------------

def _keys_lower(d: dict) -> set[str]:
    return {k.lower() for k in d}


def _looks_like_review(item: object, min_text: int = 4) -> bool:
    if not isinstance(item, dict):
        return False
    lk = _keys_lower(item)
    has_rating = bool(_RATING_KEYS & lk)
    has_text   = bool(_TEXT_KEYS   & lk)
    if not (has_rating and has_text):
        return False
    # Text phải thực sự có nội dung
    for k, v in item.items():
        if k.lower() in _TEXT_KEYS and isinstance(v, str) and len(v.strip()) >= min_text:
            return True
    return False


def _find_reviews(data: object, depth: int = 0) -> list[dict]:
    """Tìm list trông giống danh sách review trong JSON."""
    if depth > 6:
        return []
    if isinstance(data, list):
        if len(data) >= 1 and sum(_looks_like_review(x) for x in data[:5]) >= 1:
            return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        for v in data.values():
            found = _find_reviews(v, depth + 1)
            if found:
                return found
    return []


def _get_field(item: dict, keys: set[str]):
    """Lấy giá trị trường đầu tiên khớp (case-insensitive)."""
    for k, v in item.items():
        if k.lower() in keys:
            return v
    return None


def _parse_date_generic(val) -> str:
    if val is None:
        return ""
    s = str(val).strip()
    if not s:
        return ""
    # Unix timestamp
    if s.lstrip("-").isdigit():
        try:
            ts = int(s)
            if ts > 1e12:
                ts //= 1000
            return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
        except Exception:
            pass
    # ISO / date string — trả nguyên
    return s[:10] if len(s) >= 10 else s


def _normalize_generic(
    item: dict, product_url: str, product_name: str
) -> Review | None:
    try:
        rating_raw = _get_field(item, _RATING_KEYS)
        rating = 5
        if rating_raw is not None:
            try:
                rating = max(1, min(5, int(float(str(rating_raw)))))
            except (ValueError, TypeError):
                pass

        text_raw = _get_field(item, _TEXT_KEYS)
        text = " ".join(str(text_raw or "").split())

        date_raw = _get_field(item, _DATE_KEYS)
        date_str = _parse_date_generic(date_raw)

        # Images — có thể là list hoặc string
        imgs_raw = _get_field(item, _IMG_KEYS)
        image_urls: list[str] = []
        if isinstance(imgs_raw, list):
            image_urls = [str(x) for x in imgs_raw if x]
        elif isinstance(imgs_raw, str) and imgs_raw.startswith("http"):
            image_urls = [imgs_raw]

        # Review ID — bất kỳ field nào tên "id"
        review_id = str(item.get("id") or item.get("review_id") or item.get("reviewId") or "")

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
        log.debug("Skip item: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Scraper class
# ---------------------------------------------------------------------------

class GenericPlaywrightScraper:
    """Scraper tự phát hiện review API cho bất kỳ site nào."""

    SITE_NAME = "Generic"

    def __init__(
        self,
        headless: bool = False,
        max_pages: int = 30,
        delay: float   = 1.5,
    ) -> None:
        self.headless  = headless
        self.max_pages = max_pages
        self.delay     = delay

    async def run(
        self,
        url: str,
        output_path: str,
        fmt: str         = "csv",
        max_reviews: int = 3000,
    ) -> int:
        domain = url.split("/")[2] if "//" in url else url
        print(f"  [{self.SITE_NAME}] Auto-detect scraper → {domain}")
        print(f"  headless={self.headless}")

        exporter = ReviewExporter(output_path, fmt)
        raw_reviews, product_name = await self._scrape_async(url, max_reviews)

        if not raw_reviews:
            raise RuntimeError(
                f"Không phát hiện được review API trên {domain}. "
                "Cần viết scraper riêng hoặc dùng LLM agent."
            )

        batch: list[Review] = []
        for raw in raw_reviews:
            r = _normalize_generic(raw, url, product_name)
            if r and r.text:
                batch.append(r)

        total_saved = exporter.save_batch(batch) if batch else 0
        print(f"  [{self.SITE_NAME}] Đã lưu {total_saved:,} reviews")
        return total_saved

    async def _scrape_async(
        self, product_url: str, max_reviews: int
    ) -> tuple[list[dict], str]:
        from playwright.async_api import async_playwright

        all_raw: list[dict] = []
        _seen: set[str]     = set()
        product_name: str   = ""
        got_response        = asyncio.Event()
        page_num            = 1

        async def _on_response(response) -> None:
            ctype = response.headers.get("content-type", "")
            if "json" not in ctype or response.status != 200:
                return
            # Bỏ qua các response không liên quan review
            url_lower = response.url.lower()
            if any(kw in url_lower for kw in (
                "analytics", "tracking", "pixel", "beacon",
                "recommend", "banner", "ads", "suggest",
            )):
                return
            try:
                body    = await response.text()
                data    = json.loads(body)
                reviews = _find_reviews(data)
                if not reviews:
                    return
                new: list[dict] = []
                for item in reviews:
                    key = (
                        str(item.get("id") or item.get("review_id") or "")
                        or str(item)[:80]
                    )
                    if key not in _seen:
                        _seen.add(key)
                        new.append(item)
                if new:
                    all_raw.extend(new)
                    got_response.set()
                    log.info(
                        "[Generic] +%d reviews từ %s (total=%d)",
                        len(new), response.url[:60], len(all_raw)
                    )
            except Exception as exc:
                log.debug("[Generic] Parse error: %s", exc)

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
                    headless=self.headless,
                    args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
                )
                context = await browser.new_context(
                    user_agent=(
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 (KHTML, like Gecko) "
                        "Chrome/124.0.0.0 Safari/537.36"
                    ),
                    locale="vi-VN",
                    timezone_id="Asia/Ho_Chi_Minh",
                    viewport={"width": 1440, "height": 900},
                )
                await context.add_init_script(
                    "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
                )
                page = await context.new_page()
                page.on("response", _on_response)

                got_response.clear()
                await page.goto(product_url, wait_until="domcontentloaded", timeout=35_000)

                # Lấy tên sản phẩm
                try:
                    h1 = page.locator("h1").first
                    if await h1.count() > 0:
                        product_name = (await h1.inner_text()).strip()
                    if not product_name:
                        product_name = (await page.title()).split("|")[0].strip()
                except Exception:
                    pass

                # Scroll để trigger lazy-load review API
                for pct in (0.4, 0.7, 1.0):
                    await page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {pct})")
                    await page.wait_for_timeout(1200)

                try:
                    await asyncio.wait_for(_await_event(got_response), timeout=20.0)
                except asyncio.TimeoutError:
                    print("  [Generic] Không intercept được JSON review sau 20s — thử captcha check")
                    ok = await _captcha_pause_and_resume(got_response, lambda: len(all_raw) > 0)
                    if not ok or not all_raw:
                        await browser.close()
                        return [], product_name

                print(f"  [Generic] Phát hiện {len(all_raw)} reviews trang 1")

                # Phân trang generic
                while len(all_raw) < max_reviews and page_num < self.max_pages:
                    clicked = False
                    for sel in _NEXT_SELS:
                        try:
                            btn = page.locator(sel).first
                            if await btn.count() > 0 and await btn.is_enabled():
                                got_response.clear()
                                await btn.scroll_into_view_if_needed(timeout=2000)
                                await btn.click(force=True, timeout=4000)
                                try:
                                    await asyncio.wait_for(_await_event(got_response), timeout=10.0)
                                    clicked = True
                                    break
                                except asyncio.TimeoutError:
                                    pass
                        except Exception:
                            continue

                    if not clicked:
                        print("  [Generic] Không tìm được nút trang tiếp — dừng.")
                        break

                    page_num += 1
                    print(f"  [Generic] Trang {page_num}: tổng {len(all_raw):,}")
                    await page.wait_for_timeout(int(self.delay * 1000))

                await browser.close()

        except Exception as exc:
            log.error("[Generic] Lỗi: %s", exc, exc_info=True)
            print(f"  [Generic] Lỗi: {exc}")

        return all_raw, product_name


# ---------------------------------------------------------------------------
# Shared utilities
# ---------------------------------------------------------------------------

async def _await_event(event: asyncio.Event) -> None:
    while not event.is_set():
        await asyncio.sleep(0.05)
    event.clear()


async def _captcha_pause_and_resume(
    got_response: asyncio.Event,
    check_fn,
    context: str = "",
) -> bool:
    deadline = CAPTCHA_AUTO
    while deadline > 0:
        got_response.clear()
        chunk = min(CAPTCHA_CHUNK, deadline)
        try:
            await asyncio.wait_for(_await_event(got_response), timeout=chunk)
            if check_fn():
                return True
        except asyncio.TimeoutError:
            pass
        deadline -= chunk

    loop = asyncio.get_event_loop()
    while True:
        print(f"\n  [Generic] Bot-check detected. Giải captcha trong browser.")
        print("  ▸ Enter tiếp tục  |  'n' dừng")
        try:
            ans = await loop.run_in_executor(None, lambda: input("  > ").strip().lower())
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
                    return True
            except asyncio.TimeoutError:
                pass
            deadline2 -= chunk
