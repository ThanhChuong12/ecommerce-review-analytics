"""
LazadaScraper — Network Interception bằng Playwright, chỉ dùng phân trang.

Cơ chế:
  1. Đăng ký listener page.on("response") để bắt MỌI review API response
  2. Navigate + scroll → trang 1 tự động bị intercept
  3. Click nút "Tiếp theo ›" lặp lại cho đến khi hết trang hoặc đủ max_reviews

Tại sao không replay POST thủ công:
  - Lazada API yêu cầu dynamic security tokens (_m_h5_tk, sign, t)
    được tạo bởi browser JS — không thể tự sinh được từ ngoài.
  - Để browser tự click → browser tự tạo token → ta chỉ intercept response.

URL pattern:
  https://www.lazada.vn/products/{slug}-i{itemId}-s{skuId}.html

API endpoint được intercept:
  POST https://acs-m.lazada.vn/h5/mtop.lazada.review.item.getpcreviewlist/1.0/
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import re
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

from scraper.exporter import ReviewExporter
from scraper.models import Review

log = logging.getLogger("LazadaScraper")

SESSION_DIR = Path("output") / "agent_sessions"
PAGE_SIZE = 10  # fallback; actual size detected từ response đầu tiên

# Thời gian chờ (giây) mỗi chunk khi polling captcha
CAPTCHA_POLL_CHUNK = 15.0
# Thời gian tối đa chờ tự động trước khi hỏi người dùng
CAPTCHA_AUTO_WAIT = 90.0

_REVIEW_PATH = "mtop.lazada.review.item.getpcreviewlist"

# Selectors cho nút "Trang tiếp" — Lazada dùng iweb-pagination-* (không phải ant-)
_NEXT_PAGE_SELECTORS = [
    "li.iweb-pagination-next:not(.iweb-pagination-disabled) button",
    "li.iweb-pagination-next:not(.iweb-pagination-disabled)",
    "button.iweb-pagination-item-link[aria-label='next page']:not([disabled])",
    "button.iweb-pagination-item-link[aria-label*='next' i]:not([disabled])",
]

# Selectors để cuộn đến khu vực review / pagination
_REVIEW_SECTION_SELECTORS = [
    ".mod-reviews-pagination",
    ".iweb-pagination",
    ".pdp-mod-review-pagination-info",
    "[data-spm='pdp_reviews']",
    "#module_review",
    ".mod-reviews",
    "[class*='pdp-review']",
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _parse_ids(url: str) -> tuple[str, str]:
    """
    Trả về (item_id, sku_id) từ Lazada product URL.
    Pattern: /products/{slug}-i{itemId}-s{skuId}.html
    """
    path = urlparse(url).path
    m = re.search(r"-i(\d+)-s(\d+)", path)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"[/-]i(\d+)", path)
    if m:
        return m.group(1), ""
    raise ValueError(f"Không tìm thấy itemId trong URL: {url}")


def _normalize_image_url(url: str) -> str:
    if url.startswith("//"):
        return "https:" + url
    return url


def _parse_date(ts) -> str:
    try:
        if isinstance(ts, str) and len(ts) >= 10 and ts[0].isdigit():
            return ts[:10]
        if isinstance(ts, str) and any(
            kw in ts for kw in ("tuần", "ngày", "tháng", "giờ", "phút")
        ):
            return ts
        if str(ts).lstrip("-").isdigit():
            ts_int = int(ts)
            if ts_int > 1e12:
                ts_int //= 1000
            dt = datetime.fromtimestamp(ts_int, tz=timezone.utc)
            return dt.strftime("%Y-%m-%d")
    except Exception:
        pass
    return str(ts) if ts else ""


def _extract_reviews_from_payload(data: dict) -> tuple[list[dict], int]:
    """Trích (items, total) từ JSON payload của Lazada review API."""
    data2 = data.get("data") or {}
    module = data2.get("module") or {}
    items: list[dict] = module.get("reviews") or module.get("items") or []
    paging = data2.get("paging") or {}
    total = int(
        paging.get("totalItems")
        or module.get("totalCount")
        or module.get("totalitem")
        or 0
    )
    return items, total


def _normalize_review(raw: dict, product_url: str, product_name: str) -> Review | None:
    """Chuẩn hóa review từ raw dict → Review model."""
    try:
        images: list[str] = []
        for media in raw.get("mediaList") or raw.get("images") or []:
            if isinstance(media, dict):
                url = (
                    media.get("videoUrl")
                    or media.get("coverUrl")
                    or media.get("url")
                    or ""
                )
            else:
                url = str(media)
            if url:
                images.append(_normalize_image_url(url))

        ts = (
            raw.get("reviewTime")
            or raw.get("gmtCreateTime")
            or raw.get("createdTime")
            or ""
        )
        date_str = _parse_date(ts)

        text = ""
        content_list = raw.get("reviewContentList") or []
        if content_list and isinstance(content_list, list):
            text = content_list[0].get("content") or ""
        if not text:
            text = raw.get("reviewContent") or raw.get("content") or ""

        # Thay \n, \r bằng space để tránh tạo extra rows trong CSV
        clean_text = " ".join(str(text).split())

        return Review(
            text=clean_text,
            rating=int(raw.get("rating") or raw.get("score") or 0),
            date=date_str,
            image_urls=images,
            product_url=product_url,
            scraped_at=datetime.now().isoformat(),
        )
    except Exception as exc:
        log.debug("[Lazada] Skip malformed review: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main scraper class
# ---------------------------------------------------------------------------


class LazadaScraper:
    """
    Lazada scraper dùng Playwright network interception + phân trang UI.
    Interface: async run(url, output_path, fmt, max_reviews) → int.
    """

    SITE_NAME = "Lazada"

    def __init__(
        self,
        headless: bool = False,
        max_pages: int = 100,
        delay: float = 1.5,
    ):
        self.headless = headless
        self.max_pages = max_pages
        self.delay = delay
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        self._state_file = SESSION_DIR / "state_lazada.vn.json"
        self._seen: set[str] = set()

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    async def run(
        self,
        url: str,
        output_path: str,
        fmt: str = "csv",
        max_reviews: int = 3000,
    ) -> int:
        """Scrape Lazada reviews và ghi ra file. Trả về số review đã lưu."""
        print(f"  [{self.SITE_NAME}] Playwright pagination scraper")
        print(f"  headless={self.headless} | max_reviews={max_reviews:,}")

        try:
            item_id, sku_id = _parse_ids(url)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        print(f"  itemId={item_id} | skuId={sku_id}")

        exporter = ReviewExporter(output_path, fmt)
        raw_reviews = await self._scrape_async(url, max_reviews)

        # Normalize + export (không cần dedup lại vì đã xử lý ở tầng raw)
        batch: list[Review] = []
        for raw in raw_reviews:
            review = _normalize_review(raw, url, f"Lazada item {item_id}")
            if review is not None:
                batch.append(review)

        total_saved = exporter.save_batch(batch) if batch else 0
        print(f"  [{self.SITE_NAME}] Tổng cộng đã lưu: {total_saved:,} reviews")
        return total_saved

    # ------------------------------------------------------------------
    # Core Playwright scraping
    # ------------------------------------------------------------------

    async def _scrape_async(self, product_url: str, max_reviews: int) -> list[dict]:
        """
        Mở browser, intercept review API responses trong khi click phân trang.
        Trả về list raw review dicts (chưa normalize).
        """
        from playwright.async_api import async_playwright

        all_raw: list[dict] = []
        total_on_server = 0
        actual_page_size = PAGE_SIZE  # cập nhật từ response thực
        page_num = 1
        _seen_raw: set[str] = set()  # dedup ngay tại tầng raw

        # asyncio.Event báo hiệu khi có response mới về
        got_response = asyncio.Event()

        async def _on_response(response) -> None:
            nonlocal total_on_server, actual_page_size
            if _REVIEW_PATH not in response.url or response.status != 200:
                return
            try:
                body = await response.text()
                data = json.loads(body)
                items, total = _extract_reviews_from_payload(data)
                new_items = []
                for item in items:
                    # Ưu tiên dùng reviewId (stable unique) làm dedup key
                    review_id = (
                        str(item.get("reviewId") or "")
                        or str(item.get("id") or "")
                        or str(item.get("reviewid") or "")
                    )
                    if review_id:
                        raw_key = f"id:{review_id}"
                    else:
                        # Fallback: hash nội dung (kém unique hơn)
                        content_list = item.get("reviewContentList") or []
                        raw_text = (
                            content_list[0].get("content", "") if content_list
                            else item.get("reviewContent") or item.get("content") or ""
                        )
                        ts = (
                            item.get("reviewTime")
                            or item.get("gmtCreateTime")
                            or item.get("createdTime")
                            or ""
                        )
                        rating = item.get("rating") or item.get("score") or 0
                        reviewer = item.get("reviewerName") or item.get("reviewer") or ""
                        raw_key = hashlib.md5(
                            f"{str(raw_text)}|{ts}|{rating}|{reviewer}".encode()
                        ).hexdigest()

                    if raw_key not in _seen_raw:
                        _seen_raw.add(raw_key)
                        new_items.append(item)

                if new_items:
                    # Detect page size thực từ lần đầu nhận data
                    if actual_page_size == PAGE_SIZE:
                        actual_page_size = len(new_items)
                    all_raw.extend(new_items)
                    log.info(
                        "[Lazada] +%d reviews (bỏ %d trùng, tổng: %d)",
                        len(new_items), len(items) - len(new_items), len(all_raw)
                    )
                if total > total_on_server:
                    total_on_server = total
                got_response.set()
            except Exception as exc:
                log.debug("[Lazada] Response parse error: %s", exc)


        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
                    headless=self.headless,
                    args=[
                        "--disable-blink-features=AutomationControlled",
                        "--no-sandbox",
                        "--disable-dev-shm-usage",
                        "--start-maximized",
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
                    "extra_http_headers": {
                        "Accept-Language": "vi-VN,vi;q=0.9,en-US;q=0.8",
                    },
                }

                if self._state_file.exists():
                    ctx_kwargs["storage_state"] = str(self._state_file)
                    log.info("[Lazada] Reusing session")

                context = await browser.new_context(**ctx_kwargs)
                await context.add_init_script(
                    "Object.defineProperty(navigator,'webdriver',{get:()=>undefined})"
                )
                page = await context.new_page()
                page.on("response", _on_response)

                # ── Bước 1: Mở trang sản phẩm và chờ review API ───────────
                got_response.clear()
                await page.goto(product_url, wait_until="domcontentloaded", timeout=35_000)
                await page.evaluate("window.scrollBy(0, document.body.scrollHeight * 0.5)")
                await page.wait_for_timeout(2000)

                print(
                    "  [Lazada] Đang chờ review API...\n"
                    "  ↳ Nếu thấy captcha/xác thực bot, hãy giải thủ công trong browser."
                )

                def _has_real_data() -> bool:
                    """Lazada trả response với 0 items khi bị block → kiểm tra dữ liệu thực."""
                    return len(all_raw) > 0 or total_on_server > 0

                page1_ok = False

                # ── Attempt 1: scroll 50%, chờ 30s ─────────────────────────
                try:
                    await asyncio.wait_for(_await_event(got_response), timeout=30.0)
                    page1_ok = _has_real_data()
                except asyncio.TimeoutError:
                    pass

                # ── Fallback 1: scroll sâu hơn (100%), chờ 15s ─────────────
                if not page1_ok:
                    print("  [Lazada] Thử scroll sâu hơn để trigger review API...")
                    got_response.clear()
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(2000)
                    try:
                        await asyncio.wait_for(_await_event(got_response), timeout=15.0)
                        page1_ok = _has_real_data()
                    except asyncio.TimeoutError:
                        pass

                # ── Fallback 2: click tab "Đánh giá" ───────────────────────
                if not page1_ok:
                    print("  [Lazada] Thử click tab 'Đánh giá'...")
                    tab_selectors = [
                        "a:has-text('Đánh giá')",
                        "a:has-text('đánh giá')",
                        "[data-spm='tab_ratings']",
                        "li:has-text('Đánh giá')",
                        "button:has-text('Ratings')",
                        "[class*='pdp-tabs'] a",
                    ]
                    for sel in tab_selectors:
                        try:
                            el = page.locator(sel).first
                            if await el.count() > 0:
                                got_response.clear()
                                await el.click(timeout=3000)
                                await page.wait_for_timeout(1500)
                                try:
                                    await asyncio.wait_for(_await_event(got_response), timeout=10.0)
                                    if _has_real_data():
                                        page1_ok = True
                                        break
                                except asyncio.TimeoutError:
                                    pass
                        except Exception:
                            continue

                # ── Fallback 3: dừng & hỏi người dùng cho đến khi giải xong ──
                if not page1_ok:
                    page1_ok = await _captcha_pause_and_resume(
                        got_response=got_response,
                        check_fn=_has_real_data,
                        context="trang đầu tiên",
                    )
                    if not page1_ok:
                        print(
                            "  [Lazada] ❌ Người dùng bỏ qua xác thực ban đầu.\n"
                            "  Thử xóa file session: output/agent_sessions/state_lazada.vn.json"
                        )

                pg1_count = len(all_raw)
                print(
                    f"  [OK] Trang 1: +{pg1_count} reviews"
                    f" | Tổng trên server: {total_on_server:,}"
                )

                if not page1_ok:
                    await browser.close()
                    return all_raw

                # Lưu session
                try:
                    await context.storage_state(path=str(self._state_file))
                except Exception:
                    pass

                # ── Bước 2: Cuộn đến pagination, click Next lặp lại ────────
                await self._scroll_to_review_section(page)

                while len(all_raw) < max_reviews and page_num <= self.max_pages:
                    # Dừng sớm nếu đã đủ trang (dùng actual_page_size, không phải PAGE_SIZE)
                    if total_on_server > 0:
                        total_pages = (total_on_server + actual_page_size - 1) // actual_page_size
                        if page_num >= total_pages:
                            print(
                                f"  [Lazada] Đã duyệt hết {total_pages} trang "
                                f"({total_on_server} reviews)."
                            )
                            break

                    clicked = await self._click_next_page(page, got_response)
                    if not clicked:
                        print("  [Lazada] Không còn nút 'Tiếp theo' — dừng.")
                        break

                    page_num += 1
                    print(
                        f"  [OK] Trang {page_num}: tổng tạm {len(all_raw):,}"
                        f"/{total_on_server:,} reviews"
                    )
                    await page.wait_for_timeout(int(self.delay * 1000))

                await browser.close()

        except Exception as exc:
            log.error("[Lazada] Lỗi: %s", exc, exc_info=True)
            print(f"  [Lazada] ❌ {exc}")

        return all_raw

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    async def _scroll_to_review_section(self, page) -> None:
        """Cuộn đến khu vực đánh giá để pagination hiện ra."""
        for sel in _REVIEW_SECTION_SELECTORS:
            try:
                el = page.locator(sel).first
                if await el.count() > 0:
                    await el.scroll_into_view_if_needed(timeout=3000)
                    await page.wait_for_timeout(600)
                    return
            except Exception:
                continue
        # Fallback: scroll đến cuối trang
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(600)

    async def _click_next_page(self, page, got_response: asyncio.Event) -> bool:
        """
        Click nút 'Trang tiếp theo' bằng JavaScript (primary) để tránh
        vấn đề scroll/viewport khi pagination đổi dạng (ví dụ: < 1 … 5 6 7 … 8 >).
        Playwright CSS click làm fallback.
        Nếu không nhận được response sau 10s (có thể do captcha giữa chừng),
        sẽ chờ tối đa CAPTCHA_WAIT_TIMEOUT giây để user giải thủ công.
        Trả về True nếu click thành công VÀ nhận được review response mới.
        """
        clicked = await self._do_click_next(page, got_response)
        if not clicked:
            return False

        # Nhận được response bình thường → tiếp tục
        got = await self._wait_for_response_or_captcha(page, got_response)
        return got

    async def _do_click_next(self, page, got_response: asyncio.Event) -> bool:
        """
        Thực hiện click nút Next (JS primary → Playwright fallback).
        Trả về True nếu tìm thấy và click được nút, bất kể có response hay không.
        """
        # ── Primary: JavaScript click (bypass scroll & viewport) ──────────
        try:
            got_response.clear()
            was_clicked: bool = await page.evaluate("""
                () => {
                    // Lazada dùng iweb-pagination-* (không phải ant-pagination-*)
                    const candidates = [
                        'li.iweb-pagination-next:not(.iweb-pagination-disabled) button',
                        'li.iweb-pagination-next:not(.iweb-pagination-disabled)',
                        'button.iweb-pagination-item-link[aria-label="next page"]',
                        'button.iweb-pagination-item-link[aria-label*="next"]',
                    ];
                    for (const sel of candidates) {
                        const el = document.querySelector(sel);
                        if (el && !el.disabled && !el.closest('.iweb-pagination-disabled')) {
                            el.scrollIntoView({ block: 'center', behavior: 'instant' });
                            el.click();
                            return true;
                        }
                    }
                    return false;
                }
            """)
            if was_clicked:
                return True
        except Exception as exc:
            log.debug("[Lazada] JS click failed: %s", exc)

        # ── Fallback: Playwright locator click ────────────────────────────
        await self._scroll_to_review_section(page)
        for sel in _NEXT_PAGE_SELECTORS:
            try:
                btn = page.locator(sel).first
                if await btn.count() == 0:
                    continue
                if not await btn.is_enabled():
                    continue
                await btn.scroll_into_view_if_needed(timeout=2000)
                got_response.clear()
                await btn.click(force=True, timeout=4000)
                return True
            except Exception as exc:
                log.debug("[Lazada] Selector '%s' failed: %s", sel, exc)
                continue

        return False

    async def _wait_for_response_or_captcha(
        self, page, got_response: asyncio.Event
    ) -> bool:
        """
        Chờ review API response sau khi click Next.
        - Nếu nhận được trong 10s → trả về True ngay.
        - Nếu không → dừng & hỏi người dùng tương tác để tiếp tục,
          không giới hạn số lần giải captcha.
        """
        # ── Bình thường: chờ 10s ──────────────────────────────────────────
        try:
            await asyncio.wait_for(_await_event(got_response), timeout=10.0)
            return True
        except asyncio.TimeoutError:
            pass

        # ── Hết 10s mà không có response → nghi captcha ──────────────────
        # Nếu _await_event không timeout = response đã về → check_fn luôn True
        return await _captcha_pause_and_resume(
            got_response=got_response,
            check_fn=lambda: True,
            context="phân trang",
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


async def _await_event(event: asyncio.Event) -> None:
    """Chờ asyncio.Event được set (polling nhẹ 50ms)."""
    while not event.is_set():
        await asyncio.sleep(0.05)
    event.clear()


async def _captcha_pause_and_resume(
    got_response: asyncio.Event,
    check_fn,
    context: str = "phân trang",
) -> bool:
    """
    Cơ chế pause & resume khi gặp captcha / bot-check:

    1. Chờ tự động CAPTCHA_AUTO_WAIT giây (polling CAPTCHA_POLL_CHUNK một lần)
       → nếu response đến tự nhiên thì tiếp tục luôn.
    2. Nếu vẫn không có → IN THÔNG BÁO RÕ RÀNG và hỏi người dùng:
       - Nhấn Enter  → tiếp tục chờ thêm (có thể nhiều lần)
       - Nhập 'n'    → bỏ qua, dừng scraping
    Trả về True nếu cuối cùng nhận được dữ liệu, False nếu người dùng từ chối.
    """
    bar = "─" * 60
    # Chờ tự động trước khi hỏi
    deadline = CAPTCHA_AUTO_WAIT
    while deadline > 0:
        got_response.clear()
        chunk = min(CAPTCHA_POLL_CHUNK, deadline)
        try:
            await asyncio.wait_for(_await_event(got_response), timeout=chunk)
            if check_fn():
                print("  [Lazada] ✅ Tự động nhận được response — tiếp tục.")
                return True
        except asyncio.TimeoutError:
            pass
        deadline -= chunk
        if deadline > 0:
            print(f"  [Lazada] ⏳ Đang chờ tự động... còn {int(deadline)}s")

    # --- Hết chờ tự động: yêu cầu người dùng xác nhận ---
    while True:
        print(f"\n  {bar}")
        print(f"  [Lazada] 🔒 BỊ CHẶN BOT ({context})")
        print(f"  {bar}")
        print("  Lazada đã kích hoạt captcha / xác thực bot.")
        print("  👉 Hãy giải captcha trong cửa sổ browser đang mở.")
        print("  Sau khi giải xong và thấy trang load lại bình thường:")
        print("    ▸ Nhấn ENTER để tiếp tục thu thập dữ liệu")
        print("    ▸ Nhập 'n' rồi Enter để dừng và lưu dữ liệu hiện có")
        print(f"  {bar}")

        # Đọc input trong thread riêng để không block event loop
        loop = asyncio.get_event_loop()
        try:
            user_input = await loop.run_in_executor(
                None,
                lambda: input("  Bạn chọn [Enter/n]: ").strip().lower()
            )
        except (EOFError, KeyboardInterrupt):
            print("\n  [Lazada] Dừng theo yêu cầu.")
            return False

        if user_input == "n":
            print("  [Lazada] 🛑 Người dùng chọn dừng — lưu dữ liệu đã có.")
            return False

        # Người dùng nhấn Enter → chờ thêm CAPTCHA_AUTO_WAIT giây
        print(f"  [Lazada] ⏳ Đang chờ response sau khi giải captcha ({int(CAPTCHA_AUTO_WAIT)}s)...")
        deadline2 = CAPTCHA_AUTO_WAIT
        while deadline2 > 0:
            got_response.clear()
            chunk = min(CAPTCHA_POLL_CHUNK, deadline2)
            try:
                await asyncio.wait_for(_await_event(got_response), timeout=chunk)
                if check_fn():
                    print("  [Lazada] ✅ Nhận được response sau khi giải captcha — tiếp tục!")
                    return True
            except asyncio.TimeoutError:
                pass
            deadline2 -= chunk
            if deadline2 > 0:
                print(f"  [Lazada] ⏳ Vẫn chờ... còn {int(deadline2)}s")

        # Vẫn không nhận được → hỏi lại
        print("  [Lazada] ⚠️  Vẫn chưa nhận được dữ liệu sau khi giải. Thử lại?")
