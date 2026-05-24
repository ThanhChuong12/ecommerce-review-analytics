"""
LazadaScraper â€” Network Interception báº±ng Playwright, chá»‰ dÃ¹ng phÃ¢n trang.

CÆ¡ cháº¿:
  1. ÄÄƒng kÃ½ listener page.on("response") Ä‘á»ƒ báº¯t Má»ŒI review API response
  2. Navigate + scroll â†’ trang 1 tá»± Ä‘á»™ng bá»‹ intercept
  3. Click nÃºt "Tiáº¿p theo â€º" láº·p láº¡i cho Ä‘áº¿n khi háº¿t trang hoáº·c Ä‘á»§ max_reviews

Táº¡i sao khÃ´ng replay POST thá»§ cÃ´ng:
  - Lazada API yÃªu cáº§u dynamic security tokens (_m_h5_tk, sign, t)
    Ä‘Æ°á»£c táº¡o bá»Ÿi browser JS â€” khÃ´ng thá»ƒ tá»± sinh Ä‘Æ°á»£c tá»« ngoÃ i.
  - Äá»ƒ browser tá»± click â†’ browser tá»± táº¡o token â†’ ta chá»‰ intercept response.

URL pattern:
  https://www.lazada.vn/products/{slug}-i{itemId}-s{skuId}.html

API endpoint Ä‘Æ°á»£c intercept:
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
PAGE_SIZE = 10  # fallback; actual size detected tá»« response Ä‘áº§u tiÃªn

# Thá»i gian chá» (giÃ¢y) má»—i chunk khi polling captcha
CAPTCHA_POLL_CHUNK = 10.0
# Thá»i gian tá»‘i Ä‘a chá» tá»± Ä‘á»™ng trÆ°á»›c khi há»i ngÆ°á»i dÃ¹ng
CAPTCHA_AUTO_WAIT = 30.0

_REVIEW_PATH = "mtop.lazada.review.item.getpcreviewlist"

# Selectors cho nÃºt "Trang tiáº¿p" â€” Lazada dÃ¹ng iweb-pagination-* (khÃ´ng pháº£i ant-)
_NEXT_PAGE_SELECTORS = [
    "li.iweb-pagination-next:not(.iweb-pagination-disabled) button",
    "li.iweb-pagination-next:not(.iweb-pagination-disabled)",
    "button.iweb-pagination-item-link[aria-label='next page']:not([disabled])",
    "button.iweb-pagination-item-link[aria-label*='next' i]:not([disabled])",
]

# Selectors Ä‘á»ƒ cuá»™n Ä‘áº¿n khu vá»±c review / pagination
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
    Tráº£ vá» (item_id, sku_id) tá»« Lazada product URL.
    Pattern: /products/{slug}-i{itemId}-s{skuId}.html
    """
    path = urlparse(url).path
    m = re.search(r"-i(\d+)-s(\d+)", path)
    if m:
        return m.group(1), m.group(2)
    m = re.search(r"[/-]i(\d+)", path)
    if m:
        return m.group(1), ""
    raise ValueError(f"KhÃ´ng tÃ¬m tháº¥y itemId trong URL: {url}")


def _normalize_image_url(url: str) -> str:
    if url.startswith("//"):
        return "https:" + url
    return url


def _parse_date(ts) -> str:
    try:
        if isinstance(ts, str) and len(ts) >= 10 and ts[0].isdigit():
            return ts[:10]
        if isinstance(ts, str) and any(
            kw in ts for kw in ("tuáº§n", "ngÃ y", "thÃ¡ng", "giá»", "phÃºt")
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
    """TrÃ­ch (items, total) tá»« JSON payload cá»§a Lazada review API."""
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
    """Chuáº©n hÃ³a review tá»« raw dict â†’ Review model."""
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

        # Thay \n, \r báº±ng space Ä‘á»ƒ trÃ¡nh táº¡o extra rows trong CSV
        clean_text = " ".join(str(text).split())

        return Review(
            product_name=product_name,
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
    Lazada scraper dÃ¹ng Playwright network interception + phÃ¢n trang UI.
    Interface: async run(url, output_path, fmt, max_reviews) â†’ int.
    """

    SITE_NAME = "Lazada"

    def __init__(
        self,
        headless: bool = False,
        max_pages: int = 200,
        delay: float = 0.3,
    ):
        self.headless = headless
        self.max_pages = max_pages
        self.delay = delay
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        self._state_file = SESSION_DIR / "state_lazada.vn.json"
        self._seen: set[str] = set()
        self._t0: float = 0  # start time for speed calc

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
        """Scrape Lazada reviews vÃ  ghi ra file. Tráº£ vá» sá»‘ review Ä‘Ã£ lÆ°u."""
        import time
        print(f"  [{self.SITE_NAME}] CloakBrowser pagination scraper (optimized)")
        print(f"  headless={self.headless} | max_reviews={max_reviews:,} | delay={self.delay}s")

        try:
            item_id, sku_id = _parse_ids(url)
        except ValueError as exc:
            raise ValueError(str(exc)) from exc

        print(f"  itemId={item_id} | skuId={sku_id}")

        exporter = ReviewExporter(output_path, fmt)
        raw_reviews, product_name = await self._scrape_async(url, max_reviews)

        # Normalize + export (khÃ´ng cáº§n dedup láº¡i vÃ¬ Ä‘Ã£ xá»­ lÃ½ á»Ÿ táº§ng raw)
        batch: list[Review] = []
        for raw in raw_reviews:
            review = _normalize_review(raw, url, product_name)
            if review is not None:
                batch.append(review)

        total_saved = exporter.save_batch(batch) if batch else 0
        print(f"  [{self.SITE_NAME}] Tá»•ng cá»™ng Ä‘Ã£ lÆ°u: {total_saved:,} reviews")
        return total_saved

    # ------------------------------------------------------------------
    # Core scraping (CloakBrowser + retry)
    # ------------------------------------------------------------------

    async def _scrape_async(
        self, product_url: str, max_reviews: int
    ) -> tuple[list[dict], str]:
        """Retry up to 3 times with exponential backoff."""
        for attempt in range(3):
            try:
                result = await self._scrape_attempt(product_url, max_reviews)
                if result[0]:
                    return result
                if attempt < 2:
                    wait = 2 ** attempt
                    log.warning("[Lazada] Attempt %d empty, retry in %ds", attempt + 1, wait)
                    await asyncio.sleep(wait)
            except Exception as exc:
                if attempt == 2:
                    log.error("[Lazada] All 3 attempts failed: %s", exc)
                    return [], ""
                wait = 2 ** attempt
                log.warning("[Lazada] Attempt %d error: %s - retry in %ds", attempt + 1, exc, wait)
                await asyncio.sleep(wait)
        return [], ""

    async def _scrape_attempt(
        self, product_url: str, max_reviews: int
    ) -> tuple[list[dict], str]:
        """Single scrape attempt using CloakBrowser (or Playwright fallback)."""
        from scraper.stealth_browser import launch_stealth_context

        all_raw: list[dict] = []
        total_on_server = 0
        actual_page_size = PAGE_SIZE
        page_num = 1
        _seen_raw: set[str] = set()
        product_name: str = ""
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
                    review_id = (
                        str(item.get("reviewId") or "")
                        or str(item.get("id") or "")
                        or str(item.get("reviewid") or "")
                    )
                    if review_id:
                        raw_key = f"id:{review_id}"
                    else:
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
                    if actual_page_size == PAGE_SIZE:
                        actual_page_size = len(new_items)
                    all_raw.extend(new_items)
                    log.info(
                        "[Lazada] +%d reviews (dedup %d, total: %d)",
                        len(new_items), len(items) - len(new_items), len(all_raw)
                    )
                if total > total_on_server:
                    total_on_server = total
                got_response.set()
            except Exception as exc:
                log.debug("[Lazada] Response parse error: %s", exc)

        try:
            import time
            self._t0 = time.time()
            context = await launch_stealth_context(
                storage_state=str(self._state_file) if self._state_file.exists() else None,
                headless=self.headless,
                humanize=False,
            )
            page = await context.new_page()
            page.on("response", _on_response)

            got_response.clear()
            await page.goto(product_url, wait_until="domcontentloaded", timeout=35_000)

            # â”€â”€ Check for verification/captcha page â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
            # Lazada may redirect to a verification page before showing product
            page_url = page.url.lower()
            page_title = ""
            try:
                page_title = (await page.title()).lower()
            except Exception:
                pass

            is_verify_page = any(kw in page_url for kw in ["verify", "captcha", "security", "login"])
            is_verify_page = is_verify_page or any(kw in page_title for kw in ["verify", "xÃ¡c minh", "security"])

            if is_verify_page and not self.headless:
                bar = "â”€" * 60
                print(f"\n  {bar}")
                print(f"  [Lazada] ðŸ”’ TRÃŒNH DUYá»†T YÃŠU Cáº¦U XÃC THá»°C")
                print(f"  {bar}")
                print(f"  Lazada yÃªu cáº§u xÃ¡c minh báº¡n lÃ  ngÆ°á»i tháº­t.")
                print(f"  ðŸ‘‰ HÃ£y hoÃ n táº¥t xÃ¡c thá»±c trong cá»­a sá»• browser.")
                print(f"  Sau khi xong, trang sáº£n pháº©m sáº½ tá»± load láº¡i.")
                print(f"  {bar}")

                # Wait up to 120s for navigation away from verify page
                for _ in range(24):  # 24 * 5s = 120s
                    await page.wait_for_timeout(5000)
                    cur_url = page.url.lower()
                    if not any(kw in cur_url for kw in ["verify", "captcha", "security", "login"]):
                        print("  [Lazada] âœ… XÃ¡c thá»±c thÃ nh cÃ´ng! Tiáº¿p tá»¥c...")
                        await page.wait_for_timeout(2000)
                        break
                else:
                    print("  [Lazada] âš ï¸ Háº¿t thá»i gian chá» xÃ¡c thá»±c (120s).")
                    # Ask user
                    loop = asyncio.get_event_loop()
                    try:
                        user_input = await loop.run_in_executor(
                            None, lambda: input("  ÄÃ£ xÃ¡c thá»±c xong? [Enter=tiáº¿p tá»¥c / n=bá» qua]: ").strip().lower()
                        )
                    except (EOFError, KeyboardInterrupt):
                        user_input = "n"
                    if user_input == "n":
                        await context.close()
                        return [], ""

            await page.evaluate("window.scrollBy(0, document.body.scrollHeight * 0.5)")
            await page.wait_for_timeout(2000)

            try:
                h1_el = page.locator('h1').first
                if await h1_el.count() > 0:
                    product_name = (await h1_el.inner_text()).strip()
                if not product_name:
                    title = await page.title()
                    product_name = title.split('|')[0].strip()
            except Exception:
                pass

            print(
                "  [Lazada] Waiting for review API...\n"
                "  -> If captcha appears, solve it manually in the browser."
            )

            def _has_real_data() -> bool:
                return len(all_raw) > 0 or total_on_server > 0

            page1_ok = False

            try:
                await asyncio.wait_for(_await_event(got_response), timeout=30.0)
                page1_ok = _has_real_data()
            except asyncio.TimeoutError:
                pass

            if not page1_ok:
                print("  [Lazada] Deep scroll to trigger review API...")
                got_response.clear()
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                await page.wait_for_timeout(2000)
                try:
                    await asyncio.wait_for(_await_event(got_response), timeout=15.0)
                    page1_ok = _has_real_data()
                except asyncio.TimeoutError:
                    pass

            if not page1_ok:
                print("  [Lazada] Trying to click review tab...")
                for sel in ["[data-spm='tab_ratings']", "[class*='pdp-tabs'] a"]:
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

            if not page1_ok:
                page1_ok = await _captcha_pause_and_resume(
                    got_response=got_response,
                    check_fn=_has_real_data,
                    context="page 1",
                )

            print(
                f"  [OK] Page 1: +{len(all_raw)} reviews"
                f" | Server total: {total_on_server:,}"
            )

            if not page1_ok:
                await context.close()
                return all_raw, product_name

            try:
                await context.storage_state(path=str(self._state_file))
            except Exception:
                pass

            await self._scroll_to_review_section(page)

            target = max_reviews if max_reviews > 0 else total_on_server
            stale_pages = 0  # pages with 0 new reviews
            MAX_STALE_PAGES = 5  # stop after N consecutive stale pages

            while (max_reviews == 0 or len(all_raw) < max_reviews) and page_num <= self.max_pages:
                if total_on_server > 0:
                    total_pages = (total_on_server + actual_page_size - 1) // actual_page_size
                    if page_num >= total_pages:
                        print(f"  [Lazada] Reached last page {total_pages}.")
                        break

                prev_count = len(all_raw)
                clicked = await self._click_next_page(page, got_response)
                if not clicked:
                    print("  [Lazada] No more Next button.")
                    break

                page_num += 1
                new_this_page = len(all_raw) - prev_count

                # Early stop: if N consecutive pages return 0 new reviews, stop
                if new_this_page == 0:
                    stale_pages += 1
                    if stale_pages >= MAX_STALE_PAGES:
                        print(f"  [Lazada] {MAX_STALE_PAGES} pages with 0 new reviews â€” stopping.")
                        print(f"  [Lazada] Lazada caps reviews at ~{len(all_raw)} for this session.")
                        break
                else:
                    stale_pages = 0

                elapsed = time.time() - self._t0 if self._t0 else 0
                speed = len(all_raw) / elapsed if elapsed > 0 else 0
                remaining = max(0, target - len(all_raw))
                eta = remaining / speed if speed > 0 else 0
                eta_str = f"{eta/60:.1f}m" if eta > 60 else f"{eta:.0f}s"
                pct = len(all_raw) / target * 100 if target > 0 else 0
                # Only print every 5 pages to reduce noise (or always for pages with data)
                if new_this_page > 0 or page_num % 5 == 0:
                    print(
                        f"  [OK] Page {page_num}: {len(all_raw):,}/{target:,} ({pct:.1f}%)"
                        f" | {speed:.1f} rev/s | ETA {eta_str}"
                    )
                await page.wait_for_timeout(int(self.delay * 1000))

            await context.close()

        except Exception as exc:
            log.error("[Lazada] Error: %s", exc, exc_info=True)
            print(f"  [Lazada] Error: {exc}")

        return all_raw, product_name

    # ------------------------------------------------------------------
    # UI helpers
    # ------------------------------------------------------------------

    async def _scroll_to_review_section(self, page) -> None:
        """Cuá»™n Ä‘áº¿n khu vá»±c Ä‘Ã¡nh giÃ¡ Ä‘á»ƒ pagination hiá»‡n ra."""
        for sel in _REVIEW_SECTION_SELECTORS:
            try:
                el = page.locator(sel).first
                if await el.count() > 0:
                    await el.scroll_into_view_if_needed(timeout=3000)
                    await page.wait_for_timeout(300)
                    return
            except Exception:
                continue
        # Fallback: scroll Ä‘áº¿n cuá»‘i trang
        await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
        await page.wait_for_timeout(300)

    async def _click_next_page(self, page, got_response: asyncio.Event) -> bool:
        """
        Click nÃºt 'Trang tiáº¿p theo' báº±ng JavaScript (primary) Ä‘á»ƒ trÃ¡nh
        váº¥n Ä‘á» scroll/viewport khi pagination Ä‘á»•i dáº¡ng (vÃ­ dá»¥: < 1 â€¦ 5 6 7 â€¦ 8 >).
        Playwright CSS click lÃ m fallback.
        Náº¿u khÃ´ng nháº­n Ä‘Æ°á»£c response sau 10s (cÃ³ thá»ƒ do captcha giá»¯a chá»«ng),
        sáº½ chá» tá»‘i Ä‘a CAPTCHA_WAIT_TIMEOUT giÃ¢y Ä‘á»ƒ user giáº£i thá»§ cÃ´ng.
        Tráº£ vá» True náº¿u click thÃ nh cÃ´ng VÃ€ nháº­n Ä‘Æ°á»£c review response má»›i.
        """
        clicked = await self._do_click_next(page, got_response)
        if not clicked:
            return False

        # Nháº­n Ä‘Æ°á»£c response bÃ¬nh thÆ°á»ng â†’ tiáº¿p tá»¥c
        got = await self._wait_for_response_or_captcha(page, got_response)
        return got

    async def _do_click_next(self, page, got_response: asyncio.Event) -> bool:
        """
        Thá»±c hiá»‡n click nÃºt Next (JS primary â†’ Playwright fallback).
        Tráº£ vá» True náº¿u tÃ¬m tháº¥y vÃ  click Ä‘Æ°á»£c nÃºt, báº¥t ká»ƒ cÃ³ response hay khÃ´ng.
        """
        # â”€â”€ Primary: JavaScript click (bypass scroll & viewport) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            got_response.clear()
            was_clicked: bool = await page.evaluate("""
                () => {
                    // Lazada dÃ¹ng iweb-pagination-* (khÃ´ng pháº£i ant-pagination-*)
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

        # â”€â”€ Fallback: Playwright locator click â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        Chá» review API response sau khi click Next.
        - Náº¿u nháº­n Ä‘Æ°á»£c trong 10s â†’ tráº£ vá» True ngay.
        - Náº¿u khÃ´ng â†’ dá»«ng & há»i ngÆ°á»i dÃ¹ng tÆ°Æ¡ng tÃ¡c Ä‘á»ƒ tiáº¿p tá»¥c,
          khÃ´ng giá»›i háº¡n sá»‘ láº§n giáº£i captcha.
        """
        # â”€â”€ BÃ¬nh thÆ°á»ng: chá» 10s â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        try:
            await asyncio.wait_for(_await_event(got_response), timeout=5.0)
            return True
        except asyncio.TimeoutError:
            pass

        # â”€â”€ Háº¿t 10s mÃ  khÃ´ng cÃ³ response â†’ nghi captcha â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Náº¿u _await_event khÃ´ng timeout = response Ä‘Ã£ vá» â†’ check_fn luÃ´n True
        return await _captcha_pause_and_resume(
            got_response=got_response,
            check_fn=lambda: True,
            context="phÃ¢n trang",
        )


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------


async def _await_event(event: asyncio.Event) -> None:
    """Chá» asyncio.Event Ä‘Æ°á»£c set (polling nháº¹ 50ms)."""
    while not event.is_set():
        await asyncio.sleep(0.05)
    event.clear()


async def _captcha_pause_and_resume(
    got_response: asyncio.Event,
    check_fn,
    context: str = "phÃ¢n trang",
) -> bool:
    """
    CÆ¡ cháº¿ pause & resume khi gáº·p captcha / bot-check:

    1. Chá» tá»± Ä‘á»™ng CAPTCHA_AUTO_WAIT giÃ¢y (polling CAPTCHA_POLL_CHUNK má»™t láº§n)
       â†’ náº¿u response Ä‘áº¿n tá»± nhiÃªn thÃ¬ tiáº¿p tá»¥c luÃ´n.
    2. Náº¿u váº«n khÃ´ng cÃ³ â†’ IN THÃ”NG BÃO RÃ• RÃ€NG vÃ  há»i ngÆ°á»i dÃ¹ng:
       - Nháº¥n Enter  â†’ tiáº¿p tá»¥c chá» thÃªm (cÃ³ thá»ƒ nhiá»u láº§n)
       - Nháº­p 'n'    â†’ bá» qua, dá»«ng scraping
    Tráº£ vá» True náº¿u cuá»‘i cÃ¹ng nháº­n Ä‘Æ°á»£c dá»¯ liá»‡u, False náº¿u ngÆ°á»i dÃ¹ng tá»« chá»‘i.
    """
    bar = "â”€" * 60
    # Chá» tá»± Ä‘á»™ng trÆ°á»›c khi há»i
    deadline = CAPTCHA_AUTO_WAIT
    while deadline > 0:
        got_response.clear()
        chunk = min(CAPTCHA_POLL_CHUNK, deadline)
        try:
            await asyncio.wait_for(_await_event(got_response), timeout=chunk)
            if check_fn():
                print("  [Lazada] âœ… Tá»± Ä‘á»™ng nháº­n Ä‘Æ°á»£c response â€” tiáº¿p tá»¥c.")
                return True
        except asyncio.TimeoutError:
            pass
        deadline -= chunk
        if deadline > 0:
            print(f"  [Lazada] â³ Äang chá» tá»± Ä‘á»™ng... cÃ²n {int(deadline)}s")

    # --- Háº¿t chá» tá»± Ä‘á»™ng: yÃªu cáº§u ngÆ°á»i dÃ¹ng xÃ¡c nháº­n ---
    while True:
        print(f"\n  {bar}")
        print(f"  [Lazada] ðŸ”’ Bá»Š CHáº¶N BOT ({context})")
        print(f"  {bar}")
        print("  Lazada Ä‘Ã£ kÃ­ch hoáº¡t captcha / xÃ¡c thá»±c bot.")
        print("  ðŸ‘‰ HÃ£y giáº£i captcha trong cá»­a sá»• browser Ä‘ang má»Ÿ.")
        print("  Sau khi giáº£i xong vÃ  tháº¥y trang load láº¡i bÃ¬nh thÆ°á»ng:")
        print("    â–¸ Nháº¥n ENTER Ä‘á»ƒ tiáº¿p tá»¥c thu tháº­p dá»¯ liá»‡u")
        print("    â–¸ Nháº­p 'n' rá»“i Enter Ä‘á»ƒ dá»«ng vÃ  lÆ°u dá»¯ liá»‡u hiá»‡n cÃ³")
        print(f"  {bar}")

        # Äá»c input trong thread riÃªng Ä‘á»ƒ khÃ´ng block event loop
        loop = asyncio.get_event_loop()
        try:
            user_input = await loop.run_in_executor(
                None,
                lambda: input("  Báº¡n chá»n [Enter/n]: ").strip().lower()
            )
        except (EOFError, KeyboardInterrupt):
            print("\n  [Lazada] Dá»«ng theo yÃªu cáº§u.")
            return False

        if user_input == "n":
            print("  [Lazada] ðŸ›‘ NgÆ°á»i dÃ¹ng chá»n dá»«ng â€” lÆ°u dá»¯ liá»‡u Ä‘Ã£ cÃ³.")
            return False

        # NgÆ°á»i dÃ¹ng nháº¥n Enter â†’ chá» thÃªm CAPTCHA_AUTO_WAIT giÃ¢y
        print(f"  [Lazada] â³ Äang chá» response sau khi giáº£i captcha ({int(CAPTCHA_AUTO_WAIT)}s)...")
        deadline2 = CAPTCHA_AUTO_WAIT
        while deadline2 > 0:
            got_response.clear()
            chunk = min(CAPTCHA_POLL_CHUNK, deadline2)
            try:
                await asyncio.wait_for(_await_event(got_response), timeout=chunk)
                if check_fn():
                    print("  [Lazada] âœ… Nháº­n Ä‘Æ°á»£c response sau khi giáº£i captcha â€” tiáº¿p tá»¥c!")
                    return True
            except asyncio.TimeoutError:
                pass
            deadline2 -= chunk
            if deadline2 > 0:
                print(f"  [Lazada] â³ Váº«n chá»... cÃ²n {int(deadline2)}s")

        # Váº«n khÃ´ng nháº­n Ä‘Æ°á»£c â†’ há»i láº¡i
        print("  [Lazada] âš ï¸  Váº«n chÆ°a nháº­n Ä‘Æ°á»£c dá»¯ liá»‡u sau khi giáº£i. Thá»­ láº¡i?")
