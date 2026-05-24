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
        headless: bool    = False,
        humanize: bool    = True,           # CloakBrowser human-like behavior
        human_preset: str = "careful",      # 'default' | 'careful'
        max_pages: int    = 100,
        delay: float      = 0.5,
    ) -> None:
        self.headless      = headless
        self.humanize      = humanize
        self.human_preset  = human_preset
        self.max_pages     = max_pages
        self.delay         = delay

    async def run(
        self,
        url: str,
        output_path: str,
        fmt: str         = "csv",
        max_reviews: int = 3000,
    ) -> int:
        domain = url.split("/")[2] if "//" in url else url
        print(f"  [{self.SITE_NAME}] Auto-detect scraper → {domain}")
        print(f"  headless={self.headless} | humanize={self.humanize} | preset={self.human_preset}")

        exporter = ReviewExporter(output_path, fmt)
        try:
            raw_reviews, product_name = await self._scrape_async(url, max_reviews)
        except (asyncio.CancelledError, KeyboardInterrupt):
            print(f"  [{self.SITE_NAME}] Interrupted — returning 0 reviews")
            return 0

        if not raw_reviews:
            raise RuntimeError(
                f"Khong phat hien duoc review API tren {domain}. "
                "Can viet scraper rieng hoac dung LLM agent."
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
        """3-phase scraping strategy:
        Phase 1: JSON API interception (fast, structured data)
        Phase 2: CloakBrowser DOM extraction (supplement or fallback)
        Phase 3: LLM Agent — handled by dispatcher (last resort)

        Phase 2 runs if Phase 1 returns 0 OR returns fewer than max_reviews.
        Results from both phases are merged and deduplicated.
        """
        api_results: list[dict] = []
        dom_results: list[dict] = []
        product_name = ""

        # ── Phase 1: JSON API interception ───────────────────────────────
        print("  [Generic] Phase 1: JSON API interception...")
        for attempt in range(2):
            try:
                api_results, product_name = await self._scrape_attempt(product_url, max_reviews)
                if api_results:
                    print(f"  [Generic] Phase 1: {len(api_results)} reviews via API")
                    break
                if attempt == 0:
                    log.warning("[Generic] Phase 1 attempt 1 empty, retry in 2s")
                    await asyncio.sleep(2)
            except Exception as exc:
                log.warning("[Generic] Phase 1 attempt %d error: %s", attempt + 1, exc)
                await asyncio.sleep(2)

        # ── Phase 2: CloakBrowser DOM extraction ─────────────────────────
        # Trigger if: (a) Phase 1 got 0 reviews, OR (b) got < max_reviews (pagination stopped early)
        if len(api_results) < max_reviews:
            if not api_results:
                print("  [Generic] Phase 1 found nothing → Phase 2: DOM extraction...")
            else:
                print(
                    f"  [Generic] Phase 1 stopped at {len(api_results)}/{max_reviews}"
                    f" → Phase 2: DOM extraction to supplement..."
                )
            try:
                dom_results, pname2 = await self._dom_scrape(product_url, max_reviews)
                if dom_results:
                    print(f"  [Generic] Phase 2: {len(dom_results)} reviews via DOM")
                    if not product_name:
                        product_name = pname2
            except Exception as exc:
                log.warning("[Generic] Phase 2 DOM error: %s", exc)

        # ── Merge + dedup ─────────────────────────────────────────────────
        if not api_results and not dom_results:
            print("  [Generic] Both phases found nothing → dispatcher will try LLM Agent.")
            return [], product_name

        # Combine, dedup by review ID first, then fall back to text
        seen_keys: set[str] = set()
        merged: list[dict] = []
        for item in api_results + dom_results:
            # Primary key: use review ID if available
            rid = str(
                item.get("id") or item.get("review_id") or item.get("reviewId") or ""
            ).strip()
            if rid:
                key = f"id:{rid}"
            else:
                # Fallback: use text content (different reviews may share same text)
                text_val = (
                    item.get("comment") or item.get("content") or item.get("text")
                    or item.get("review_content") or item.get("reviewContent")
                    or item.get("body") or ""
                )
                key = f"txt:{str(text_val).strip()[:120]}"
                if not key or key == "txt:":
                    key = f"hash:{hash(str(sorted(item.items())))}"
            if key not in seen_keys:
                seen_keys.add(key)
                merged.append(item)

        print(f"  [Generic] Merged: {len(merged)} unique reviews (API={len(api_results)}, DOM={len(dom_results)})")
        return merged[:max_reviews], product_name


    async def _scrape_attempt(
        self, product_url: str, max_reviews: int
    ) -> tuple[list[dict], str]:
        """Single scrape attempt using CloakBrowser — optimized v2.
        
        Optimizations vs v1:
        - Scroll wait 1000→500ms (saves 2s/probe)
        - Probe timeout 15→8s (faster path rejection)
        - Pagination response timeout 10→5s
        - Early-stop after 3 stale pages
        - Captured API URL for potential direct replay
        - Smart path ordering (VN vs EN domains)
        - Progress + speed logging
        """
        import time
        from scraper.stealth_browser import launch_stealth_context

        all_raw: list[dict] = []
        _seen: set[str]     = set()
        product_name: str   = ""
        got_response        = asyncio.Event()
        page_num            = 1
        _captured_api_url: str = ""  # save detected API URL for replay

        async def _on_response(response) -> None:
            nonlocal _captured_api_url
            ctype = response.headers.get("content-type", "")
            if "json" not in ctype or response.status != 200:
                return
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
                    if not _captured_api_url:
                        _captured_api_url = response.url
                    got_response.set()
                    log.info(
                        "[Generic] +%d reviews from %s (total=%d)",
                        len(new), response.url[:60], len(all_raw)
                    )
            except Exception as exc:
                log.debug("[Generic] Parse error: %s", exc)

        # Smart path ordering: detect VN vs EN domain
        domain = product_url.split("/")[2] if "//" in product_url else ""
        is_vn = domain.endswith(".vn") or ".vn/" in product_url
        if is_vn:
            _REVIEW_PATHS = [
                "",             # main page first
                "/danh-gia",    # VN priority
                "/nhan-xet",
                "/binh-luan",
                "/reviews",
            ]
        else:
            _REVIEW_PATHS = [
                "",
                "/reviews",
                "/review",
                "/feedback",
                "/comments",
            ]

        try:
            context = await launch_stealth_context(
                headless=self.headless,
                humanize=self.humanize,
                human_preset=self.human_preset,
            )
            page = await context.new_page()
            page.on("response", _on_response)

            base_url = product_url.rstrip("/")

            for path_suffix in _REVIEW_PATHS:
                target_url = base_url + path_suffix
                got_response.clear()

                print(f"  [Generic] Trying: ...{target_url[-60:]}")
                try:
                    await page.goto(target_url, wait_until="domcontentloaded", timeout=20_000)
                except Exception as e:
                    log.debug("[Generic] goto failed for %s: %s", target_url, e)
                    continue

                # Get product name from main page
                if not product_name and not path_suffix:
                    try:
                        h1 = page.locator("h1").first
                        if await h1.count() > 0:
                            product_name = (await h1.inner_text()).strip()
                        if not product_name:
                            product_name = (await page.title()).split("|")[0].strip()
                    except Exception:
                        pass

                # Scroll to trigger lazy-load review API calls (optimized: 500ms/step)
                for pct in (0.4, 0.7, 1.0):
                    await page.evaluate(
                        f"window.scrollTo(0, document.body.scrollHeight * {pct})"
                    )
                    await page.wait_for_timeout(500)

                # Wait up to 8s for a JSON review response (was 15s)
                try:
                    await asyncio.wait_for(_await_event(got_response), timeout=8.0)
                except asyncio.TimeoutError:
                    pass

                if all_raw:
                    print(f"  [Generic] Found {len(all_raw)} reviews on '{path_suffix or '/'}'")
                    if _captured_api_url:
                        print(f"  [Generic] API detected: {_captured_api_url[:80]}...")
                    break
                else:
                    log.debug("[Generic] No reviews at %s", target_url)

            if not all_raw:
                print("  [Generic] No JSON review API found on any URL — giving up.")
                # Save context for Phase 2 reuse
                self._reuse_context = context
                self._reuse_page = page
                return [], product_name

            print(f"  [Generic] Detected {len(all_raw)} reviews (page 1)")

            # ── Paginate: API replay (fast) → click fallback (slow) ──────
            t0 = time.time()
            stale_pages = 0
            MAX_STALE = 3

            # Try API URL replay first (much faster than clicking)
            replay_success = False
            if _captured_api_url:
                replay_success = await self._api_replay_paginate(
                    page, _captured_api_url, all_raw, _seen,
                    max_reviews, t0, got_response,
                )

            # Fallback: click-based pagination
            if not replay_success:
                while (max_reviews == 0 or len(all_raw) < max_reviews) and page_num < self.max_pages:
                    prev_count = len(all_raw)
                    clicked = False

                    for sel in _NEXT_SELS:
                        try:
                            btn = page.locator(sel).first
                            if await btn.count() > 0 and await btn.is_enabled():
                                got_response.clear()
                                await btn.scroll_into_view_if_needed(timeout=2000)
                                await btn.click(force=True, timeout=4000)
                                try:
                                    await asyncio.wait_for(_await_event(got_response), timeout=5.0)
                                    clicked = True
                                    break
                                except asyncio.TimeoutError:
                                    pass
                        except Exception:
                            continue

                    if not clicked:
                        print("  [Generic] No next-page button found — stopping.")
                        break

                    page_num += 1
                    new_this_page = len(all_raw) - prev_count

                    # Early-stop: 3 consecutive stale pages
                    if new_this_page == 0:
                        stale_pages += 1
                        if stale_pages >= MAX_STALE:
                            print(f"  [Generic] {MAX_STALE} stale pages — stopping at {len(all_raw)} reviews.")
                            break
                    else:
                        stale_pages = 0

                    # Progress with speed
                    elapsed = time.time() - t0
                    speed = len(all_raw) / elapsed if elapsed > 0 else 0
                    print(
                        f"  [Generic] Page {page_num}: {len(all_raw):,} reviews"
                        f" | {speed:.1f} rev/s"
                    )
                    await page.wait_for_timeout(int(self.delay * 1000))

            await context.close()

        except Exception as exc:
            log.error("[Generic] Error: %s", exc, exc_info=True)
            print(f"  [Generic] Error: {exc}")

        return all_raw, product_name


    # ------------------------------------------------------------------
    # API replay pagination — CloakBrowser JS fetch (fast path)
    # ------------------------------------------------------------------

    async def _api_replay_paginate(
        self,
        page,
        api_url: str,
        all_raw: list[dict],
        _seen: set[str],
        max_reviews: int,
        t0: float,
        got_response,
    ) -> bool:
        """Replay captured review API URL with incremented page/offset params.

        Detects common pagination patterns in the API URL:
        - page=1 → page=2, page=3, ...
        - offset=0 → offset=20, offset=40, ...
        - pageIndex=0 → pageIndex=1, ...

        Returns True if replay worked (got >0 extra reviews), False to fall back
        to click-based pagination.
        """
        import time
        from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

        parsed = urlparse(api_url)
        params = parse_qs(parsed.query, keep_blank_values=True)

        # Detect pagination parameter
        page_param = None
        offset_param = None
        page_size = None

        for key in ("page", "pageIndex", "pageNum", "p", "pageNo", "page_num"):
            if key in params:
                page_param = key
                break

        for key in ("offset", "skip", "start", "from", "cursor", "after"):
            if key in params:
                offset_param = key
                break

        for key in ("limit", "pageSize", "page_size", "size", "per_page", "count"):
            if key in params:
                try:
                    page_size = int(params[key][0])
                except (ValueError, IndexError):
                    pass
                break

        if not page_param and not offset_param:
            print("  [Generic Replay] No page/offset param in API URL — skip replay.")
            return False

        if not page_size:
            page_size = len(all_raw) if all_raw else 20

        param_name = page_param or offset_param
        param_val = params.get(param_name, ["?"])[0]
        print(f"  [Generic Replay] 🚀 API replay mode — {param_name}={param_val}, size={page_size}")

        stale_count = 0
        page_num = 1

        for attempt_page in range(2, self.max_pages + 1):
            new_params = {k: v[0] for k, v in params.items()}

            if page_param:
                try:
                    base_val = int(params[page_param][0])
                    new_params[page_param] = str(base_val + attempt_page - 1)
                except ValueError:
                    new_params[page_param] = str(attempt_page)
            elif offset_param:
                new_params[offset_param] = str(page_size * (attempt_page - 1))

            new_query = urlencode(new_params)
            new_url = urlunparse(parsed._replace(query=new_query))

            # JS fetch inside CloakBrowser — inherits all cookies/headers
            js_code = """
            async (url) => {
                try {
                    const r = await fetch(url);
                    if (!r.ok) return { status: r.status, data: null };
                    const data = await r.json();
                    return { status: r.status, data };
                } catch(e) {
                    return { status: 0, error: e.message };
                }
            }
            """

            try:
                result = await page.evaluate(js_code, new_url)
            except Exception as exc:
                log.debug("[Generic Replay] JS fetch error: %s", exc)
                break

            if not result or result.get("status") != 200 or not result.get("data"):
                stale_count += 1
                if stale_count >= 3:
                    break
                continue

            reviews = _find_reviews(result["data"])
            if not reviews:
                stale_count += 1
                if stale_count >= 3:
                    print(f"  [Generic Replay] 3 empty responses — stopping.")
                    break
                continue

            new_items: list[dict] = []
            for item in reviews:
                key = (
                    str(item.get("id") or item.get("review_id") or "")
                    or str(item)[:80]
                )
                if key not in _seen:
                    _seen.add(key)
                    new_items.append(item)

            if not new_items:
                stale_count += 1
                if stale_count >= 3:
                    print(f"  [Generic Replay] 3 stale pages — stopping.")
                    break
                continue

            stale_count = 0
            all_raw.extend(new_items)
            page_num += 1

            elapsed = time.time() - t0
            speed = len(all_raw) / elapsed if elapsed > 0 else 0
            print(
                f"  [Generic Replay] Page {page_num}: +{len(new_items)} → {len(all_raw):,}"
                f" | {speed:.1f} rev/s"
            )

            if max_reviews > 0 and len(all_raw) >= max_reviews:
                print(f"  [Generic Replay] Target {max_reviews} reached.")
                break

            await page.wait_for_timeout(int(self.delay * 1000))

        replay_got = page_num > 1
        if replay_got:
            print(f"  [Generic Replay] ✅ Total: {len(all_raw):,} reviews via API replay")
        return replay_got

    # ------------------------------------------------------------------
    # Phase 2: DOM extraction — read review elements directly from HTML
    # ------------------------------------------------------------------

    async def _dom_scrape(
        self, product_url: str, max_reviews: int
    ) -> tuple[list[dict], str]:
        """Extract reviews by reading HTML DOM directly via CloakBrowser.

        Used when no JSON API is found in Phase 1. Looks for common review
        element patterns across Vietnamese ecommerce sites.
        
        Optimized v2: reuse browser from Phase 1, faster scroll/wait times.
        """
        from scraper.stealth_browser import launch_stealth_context

        # Smart path ordering (same as Phase 1)
        domain = product_url.split("/")[2] if "//" in product_url else ""
        is_vn = domain.endswith(".vn") or ".vn/" in product_url
        if is_vn:
            _REVIEW_PATHS = ["", "/danh-gia", "/nhan-xet", "/binh-luan", "/reviews"]
        else:
            _REVIEW_PATHS = ["", "/reviews", "/review", "/feedback", "/comments"]

        # JS to extract all review-like elements from DOM
        _DOM_EXTRACT_JS = """
        () => {
            const ratingSelectors = [
                '[class*="review-item"]', '[class*="review_item"]',
                '[class*="comment-item"]', '[class*="comment_item"]',
                '[class*="rating-item"]', '[class*="danh-gia"]',
                '[class*="feedback-item"]', '[class*="nhan-xet"]',
                '[itemprop="review"]', '[data-review]',
            ];

            let items = [];
            for (const sel of ratingSelectors) {
                const els = document.querySelectorAll(sel);
                if (els.length >= 1) {
                    items = Array.from(els);
                    break;
                }
            }

            if (items.length === 0) return [];

            return items.map(el => {
                // Extract star rating
                const stars = el.querySelectorAll(
                    '[class*="star"][class*="active"], [class*="star"][class*="fill"], ' +
                    'i.fas.fa-star, i.fa-star.checked, svg[class*="star"]'
                ).length;

                // Extract text
                const textEl = el.querySelector(
                    '[class*="content"], [class*="text"], [class*="body"], ' +
                    '[class*="comment"], p, .review-text'
                );
                const text = textEl ? textEl.innerText.trim() : el.innerText.trim().slice(0, 500);

                // Extract author
                const authorEl = el.querySelector(
                    '[class*="author"], [class*="user"], [class*="name"], strong, b'
                );
                const author = authorEl ? authorEl.innerText.trim() : '';

                // Extract date
                const dateEl = el.querySelector(
                    '[class*="date"], [class*="time"], time, [datetime]'
                );
                const date = dateEl
                    ? (dateEl.getAttribute('datetime') || dateEl.innerText.trim())
                    : '';

                return { rating: stars || null, comment: text, author_name: author, date };
            }).filter(r => r.comment && r.comment.length > 3);
        }
        """;

        _NEXT_BTN_JS = """
        () => {
            const sels = [
                'a[rel="next"]', 'button[aria-label*="next" i]',
                '[class*="pagination"] [class*="next"]:not([disabled])',
                '[class*="page"] a:last-child', 'li.next a', '.pager-next a',
            ];
            for (const s of sels) {
                const el = document.querySelector(s);
                if (el && !el.disabled && el.offsetParent !== null) {
                    el.click();
                    return true;
                }
            }
            return false;
        }
        """;

        all_raw: list[dict] = []
        product_name = ""

        try:
            # ── Reuse browser from Phase 1 if available ──────────────────
            reuse = hasattr(self, "_reuse_context") and self._reuse_context
            if reuse:
                context = self._reuse_context
                page = self._reuse_page
                self._reuse_context = None
                self._reuse_page = None
                print("  [Generic DOM] Reusing browser from Phase 1")
            else:
                context = await launch_stealth_context(
                    headless=self.headless,
                    humanize=self.humanize,
                    human_preset=self.human_preset,
                )
                page = await context.new_page()

            base_url = product_url.rstrip("/")
            found_url = None

            for suffix in _REVIEW_PATHS:
                target = base_url + suffix
                try:
                    await page.goto(target, wait_until="domcontentloaded", timeout=20_000)
                except Exception:
                    continue

                if not product_name and not suffix:
                    try:
                        h1 = page.locator("h1").first
                        if await h1.count() > 0:
                            product_name = (await h1.inner_text()).strip()
                        if not product_name:
                            product_name = (await page.title()).split("|")[0].strip()
                    except Exception:
                        pass

                # Scroll to reveal lazy-loaded elements (optimized: 400ms/step)
                for pct in (0.4, 0.7, 1.0):
                    await page.evaluate(
                        f"window.scrollTo(0, document.body.scrollHeight * {pct})"
                    )
                    await page.wait_for_timeout(400)

                items = await page.evaluate(_DOM_EXTRACT_JS)
                if items:
                    all_raw.extend(items)
                    found_url = target
                    print(
                        f"  [Generic DOM] Found {len(items)} review elements on '{suffix or '/'}'"
                    )
                    break

            if not all_raw or not found_url:
                await context.close()
                return [], product_name

            # Paginate via DOM clicks (optimized waits + early-stop)
            page_num = 1
            stale_pages = 0
            while (max_reviews == 0 or len(all_raw) < max_reviews) and page_num < self.max_pages:
                await page.wait_for_timeout(int(self.delay * 1000))
                clicked = await page.evaluate(_NEXT_BTN_JS)
                if not clicked:
                    print("  [Generic DOM] No next button — stopping.")
                    break
                await page.wait_for_timeout(800)  # was 1500ms

                # Scroll to load new items (optimized: 400ms/step)
                for pct in (0.4, 0.7, 1.0):
                    await page.evaluate(
                        f"window.scrollTo(0, document.body.scrollHeight * {pct})"
                    )
                    await page.wait_for_timeout(400)

                new_items = await page.evaluate(_DOM_EXTRACT_JS)
                if not new_items:
                    stale_pages += 1
                    if stale_pages >= 3:
                        print("  [Generic DOM] 3 empty pages — stopping.")
                        break
                    continue

                # Deduplicate by text content
                existing_texts = {r.get("comment", "") for r in all_raw}
                fresh = [r for r in new_items if r.get("comment") not in existing_texts]
                if not fresh:
                    stale_pages += 1
                    if stale_pages >= 3:
                        print("  [Generic DOM] 3 stale pages — stopping.")
                        break
                    continue

                stale_pages = 0
                all_raw.extend(fresh)
                page_num += 1
                print(f"  [Generic DOM] Page {page_num}: +{len(fresh)} (total {len(all_raw)})")

            await context.close()

        except Exception as exc:
            log.error("[Generic DOM] Error: %s", exc, exc_info=True)
            print(f"  [Generic DOM] Error: {exc}")

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
    headless: bool = True,
) -> bool:
    """Wait for API response. Give up after timeout in headless/unattended mode."""
    auto_wait = 15.0  # seconds to auto-wait
    deadline = auto_wait
    while deadline > 0:
        got_response.clear()
        chunk = min(5.0, deadline)
        try:
            await asyncio.wait_for(_await_event(got_response), timeout=chunk)
            if check_fn():
                return True
        except asyncio.TimeoutError:
            pass
        deadline -= chunk

    if headless:
        print(f"  [Generic] Headless timeout — giving up ({context}).")
        return False

    # Headed mode: ask user once, then give up
    print(f"\n  [Generic] Bot-check detected. Solve captcha in browser.")
    print("  > Press ENTER to retry  |  'n' to stop")
    loop = asyncio.get_event_loop()
    try:
        ans = await loop.run_in_executor(None, lambda: input("  > ").strip().lower())
    except (EOFError, KeyboardInterrupt):
        return False
    if ans == "n":
        return False

    # One more wait after user interaction
    deadline2 = auto_wait
    while deadline2 > 0:
        got_response.clear()
        chunk = min(5.0, deadline2)
        try:
            await asyncio.wait_for(_await_event(got_response), timeout=chunk)
            if check_fn():
                return True
        except asyncio.TimeoutError:
            pass
        deadline2 -= chunk
    return False
