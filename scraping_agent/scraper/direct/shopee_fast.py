"""
shopee_fast.py — ShopeeParallelScraper v6 (CloakBrowser + JS fetch)

Architecture:
  Phase 1 — CloakBrowser warm-up (~10s):
      Open product page → intercept get_ratings → extract total + per-star counts.
      KEEP BROWSER OPEN for Phase 2.

  Phase 2 — Browser JS parallel fetch:
      Use page.evaluate(fetch()) to call Shopee API from INSIDE the browser.
      This bypasses IP bans because requests go through CloakBrowser's session.
      Batch N requests per evaluate call using Promise.allSettled().
      Each star rating (1-5) is fetched independently → bypass Shopee's 3K cap on type=0.

  Why browser fetch instead of httpx:
    - CloakBrowser passes all anti-bot checks
    - Browser shares cookies/headers/signed tokens automatically
    - No IP ban issues (requests come from the browser context)
    - Still fast: 5 parallel fetch() calls per batch, ~0.5s per batch
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse
import re

from scraper.exporter import ReviewExporter
from scraper.models import Review

# Fix Windows console encoding
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8', errors='replace')
        sys.stderr.reconfigure(encoding='utf-8', errors='replace')
    except Exception:
        pass

log = logging.getLogger("ShopeeParallelScraper")

# ── Constants ────────────────────────────────────────────────────────────────
SESSION_DIR       = Path("output") / "agent_sessions"
PROFILE_DIR       = SESSION_DIR / "profile_shopee"
CHECKPOINT_DIR    = Path("output") / "checkpoints"
COOKIES_FILE      = SESSION_DIR / "cookies_shopee.json"
_RATINGS_PATH     = "get_ratings"

DEFAULT_API_LIMIT    = 59   # Shopee accepts up to 60; use 59 to stay safe
DEFAULT_BATCH_SIZE   = 5    # parallel fetch() calls per page.evaluate
MONITOR_INTERVAL     = 10.0
BUFFER_SIZE          = 500  # flush to disk every N reviews
STAR_TYPES           = [1, 2, 3, 4, 5]
MAX_RETRIES          = 3

# Shopee API filter values (combine with star types for max coverage)
FILTER_MODES = {
    "all":     [0],           # filter=0: all reviews
    "comment": [1],           # filter=1: with comments only
    "media":   [3],           # filter=3: with images/video only
    "max":     [0, 1, 3],     # all three → 3x coverage (dedup by cmtid)
}
DEFAULT_FILTER_MODE = "max"   # maximize reviews by default

_REVIEW_SECTION = [
    ".shopee-product-rating",
    "[class*='product-rating']",
    ".page-product__reviews",
    "[class*='review']",
    ".shopee-page-controller",
]

# Shopee API base (v2)
_API_BASE = "https://shopee.vn/api/v2/item/get_ratings"


# ── Helpers ──────────────────────────────────────────────────────────────────

def _parse_ids(url: str) -> tuple[str, str]:
    """Extract (shop_id, item_id) from Shopee product URL."""
    m = re.search(r"i\.(\d+)\.(\d+)", url)
    if m:
        return m.group(1), m.group(2)
    path = urlparse(url).path
    m = re.search(r"-i\.(\d+)\.(\d+)", path) or re.search(r"i\.(\d+)\.(\d+)", path)
    if m:
        return m.group(1), m.group(2)
    parts = path.rstrip("/").split(".")
    if len(parts) >= 2 and parts[-1].isdigit() and parts[-2].isdigit():
        return parts[-2], parts[-1]
    raise ValueError(f"Cannot parse shop_id/item_id from: {url}")


def _normalize(raw: dict, product_url: str, product_name: str) -> Review | None:
    """Convert raw Shopee rating dict → Review model."""
    try:
        images = []
        for img in raw.get("images") or []:
            if img:
                images.append(f"https://cf.shopee.vn/file/{img}")
        for vid in raw.get("videos") or []:
            cover = vid.get("cover") if isinstance(vid, dict) else None
            if cover:
                images.append(f"https://cf.shopee.vn/file/{cover}")

        ts = raw.get("ctime") or raw.get("mtime") or 0
        date_str = ""
        if ts:
            try:
                date_str = datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
            except Exception:
                pass

        comment = raw.get("comment") or ""
        clean_text = " ".join(str(comment).split())

        return Review(
            review_id=str(raw.get("cmtid") or raw.get("itemid", "") + str(raw.get("ctime", ""))),
            product_name=product_name,
            text=clean_text,
            rating=int(raw.get("rating_star") or raw.get("rating") or 5),
            date=date_str,
            image_urls=images,
            product_url=product_url,
            scraped_at=datetime.now().isoformat(),
        )
    except Exception as exc:
        log.debug("Skip malformed review: %s", exc)
        return None


# ── Lightweight Checkpoint ───────────────────────────────────────────────────

class _Checkpoint:
    def __init__(self, item_id: str) -> None:
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        self._path = CHECKPOINT_DIR / f"shopee_{item_id}.json"
        self._tmp  = self._path.with_suffix(".tmp.json")

    def load(self) -> dict:
        if not self._path.exists():
            return {}
        try:
            return json.loads(self._path.read_text(encoding="utf-8"))
        except Exception:
            return {}

    def save(self, data: dict) -> None:
        try:
            self._tmp.write_text(
                json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8",
            )
            self._tmp.replace(self._path)
        except Exception as exc:
            log.warning("Checkpoint save failed: %s", exc)

    def delete(self) -> None:
        try:
            self._path.unlink(missing_ok=True)
        except Exception:
            pass


# ── JavaScript for batch fetch inside CloakBrowser ───────────────────────────

_BATCH_FETCH_JS = """
async (tasks) => {
    // tasks = [{shopid, itemid, type, offset, limit, filter}, ...]
    const results = await Promise.allSettled(
        tasks.map(async (t) => {
            const url = new URL('https://shopee.vn/api/v2/item/get_ratings');
            url.searchParams.set('filter', String(t.filter || 0));
            url.searchParams.set('flag', '1');
            url.searchParams.set('shopid', t.shopid);
            url.searchParams.set('itemid', t.itemid);
            url.searchParams.set('type', String(t.type));
            url.searchParams.set('offset', String(t.offset));
            url.searchParams.set('limit', String(t.limit));
            url.searchParams.set('exclude_filter', '1');
            const resp = await fetch(url.toString(), {
                credentials: 'include',
                headers: {
                    'Accept': 'application/json',
                    'x-api-source': 'pc',
                    'x-shopee-language': 'vi',
                    'x-requested-with': 'XMLHttpRequest',
                }
            });
            if (!resp.ok) return {status: resp.status, ratings: [], error: true};
            const data = await resp.json();
            const ratings = (data.data && data.data.ratings) || [];
            const summary = (data.data && data.data.item_rating_summary) || {};
            return {
                status: resp.status,
                ratings: ratings,
                total: summary.rating_total || 0,
                star_counts: summary.rating_count || [],
            };
        })
    );
    return results.map(r => r.status === 'fulfilled' ? r.value : {error: true, ratings: []});
}
"""

_COUNT_JS = """
async (args) => {
    const {shopid, itemid} = args;
    try {
        const url = `https://shopee.vn/api/v2/item/get_ratings?filter=0&flag=1&itemid=${itemid}&limit=1&offset=0&shopid=${shopid}&type=0`;
        const resp = await fetch(url, {
            credentials: 'include',
            headers: {
                'Accept': 'application/json',
                'x-api-source': 'pc',
                'x-shopee-language': 'vi',
                'x-requested-with': 'XMLHttpRequest',
            }
        });
        if (!resp.ok) return {error: 'HTTP ' + resp.status};
        const data = await resp.json();
        const summary = (data.data && data.data.item_rating_summary) || {};
        return {
            total: summary.rating_total || 0,
            counts: summary.rating_count || [0,0,0,0,0,0],
        };
    } catch(e) {
        return {error: e.message};
    }
}
"""


# ── Main Scraper ──────────────────────────────────────────────────────────────

class ShopeeParallelScraper:
    """
    High-speed Shopee scraper v6: CloakBrowser + browser JS fetch.
    Uses page.evaluate(fetch()) to call API through the browser session,
    bypassing IP bans while maintaining high throughput.
    """

    SITE_NAME              = "Shopee"
    BROWSER_ONLY_THRESHOLD = 200

    def __init__(
        self,
        concurrency:  int   = DEFAULT_BATCH_SIZE,
        api_limit:    int   = DEFAULT_API_LIMIT,
        headless:     bool  = False,
        humanize:     bool  = True,
        human_preset: str   = "careful",
        filter_mode:  str   = DEFAULT_FILTER_MODE,
    ) -> None:
        self.batch_size   = concurrency  # reuse param name from dispatcher
        self.api_limit    = api_limit
        self.headless     = headless
        self.humanize     = humanize
        self.human_preset = human_preset
        self.filters      = FILTER_MODES.get(filter_mode, FILTER_MODES["max"])
        self.filter_mode  = filter_mode
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        self._state_file = SESSION_DIR / "state_shopee.vn.json"

    # ── Public entry point ────────────────────────────────────────────────

    async def run(
        self,
        url:         str,
        output_path: str,
        fmt:         str = "csv",
        max_reviews: int = 0,
    ) -> int:
        shop_id, item_id = _parse_ids(url)
        checkpoint = _Checkpoint(item_id)

        filter_names = {0: 'all', 1: 'comment', 3: 'media'}
        filter_desc = '+'.join(filter_names.get(f, str(f)) for f in self.filters)
        print(f"\n  [{self.SITE_NAME}] ShopeeParallelScraper v6 (CloakBrowser + JS fetch)")
        print(f"  shopId={shop_id} | itemId={item_id}")
        print(f"  batch_size={self.batch_size} | limit={self.api_limit}/req | filters={filter_desc}")

        # ── Phase 1: CloakBrowser warm-up (keep browser open) ────────────
        print("\n  [Phase 1] CloakBrowser warm-up...")
        warmup = await self._browser_warmup(url)
        product_name, total_count, star_counts, page, context = warmup

        if page is None:
            print("  [Phase 1 FAILED] CloakBrowser warmup failed — cannot scrape")
            raise RuntimeError(
                "Shopee CloakBrowser warmup failed. Check if cloakbrowser is installed "
                "and Shopee is accessible."
            )

        print(f"  [Phase 1 OK] total={total_count:,} | product: {product_name[:60]!r}")

        # If interception didn't capture star_counts, fetch via JS
        if not star_counts and total_count > 0:
            try:
                result = await page.evaluate(_COUNT_JS, {"shopid": shop_id, "itemid": item_id})
                if not result.get("error"):
                    rc = result.get("counts") or []
                    if len(rc) >= 6:
                        star_counts = {1: rc[1], 2: rc[2], 3: rc[3], 4: rc[4], 5: rc[5]}
                        print(f"  [Phase 1] Got star counts via JS fetch")
            except Exception:
                pass

        if star_counts:
            for star in range(1, 6):
                cnt = star_counts.get(star, 0)
                reqs = (cnt + self.api_limit - 1) // self.api_limit
                print(f"    {star}*: {cnt:>6,} reviews ({reqs} requests)")

        if total_count == 0:
            print("  No reviews.")
            await context.close()
            return 0

        # ── Load checkpoint ───────────────────────────────────────────────
        ckpt = checkpoint.load()
        start_progress: dict[str, int] = {}
        if ckpt and ckpt.get("item_id") == item_id:
            start_progress = ckpt.get("progress", {})
            prev_count = ckpt.get("reviews_count", 0)
            if prev_count > 0:
                print(f"  [Checkpoint] Resume from {prev_count:,} reviews")
            else:
                start_progress = {}

        # ── Phase 2: Browser JS parallel fetch ────────────────────────────
        target = total_count if max_reviews == 0 else min(total_count, max_reviews)

        exporter = ReviewExporter(output_path, fmt)
        buffer: list[Review] = []
        buffer_lock = asyncio.Lock()
        total_saved = ckpt.get("reviews_count", 0) if ckpt else 0
        count = 0
        seen_ids: set[str] = set()  # dedup across stars
        progress = {str(s): int(start_progress.get(str(s), 0)) for s in STAR_TYPES}
        stars_done: set[int] = set()
        t0 = time.time()

        n_workers = len(STAR_TYPES) * len(self.filters)
        print(f"\n  [Phase 2] {n_workers} workers (5 stars × {len(self.filters)} filters) | batch={self.batch_size}")
        print(f"  Target: {target:,} reviews\n")

        async def _fetch_star(star: int, api_filter: int = 0) -> None:
            """Fetch all reviews for one star rating + filter, stop when API returns empty."""
            label = f"{star}*f{api_filter}"
            nonlocal count, total_saved

            star_max = star_counts.get(star, 0) if star_counts else total_count
            offset = int(progress.get(str(star), 0))
            consecutive_empty = 0

            while offset < star_max:
                # Global stop check
                if max_reviews > 0 and count >= max_reviews:
                    break

                # Build batch of requests for this star
                batch_tasks = []
                for _ in range(self.batch_size):
                    if offset >= star_max:
                        break
                    batch_tasks.append({
                        "shopid": shop_id,
                        "itemid": item_id,
                        "type": star,
                        "offset": offset,
                        "limit": self.api_limit,
                        "filter": api_filter,
                    })
                    offset += self.api_limit

                if not batch_tasks:
                    break

                # Execute batch
                for attempt in range(MAX_RETRIES):
                    try:
                        results = await page.evaluate(_BATCH_FETCH_JS, batch_tasks)
                        break
                    except Exception as exc:
                        if attempt == MAX_RETRIES - 1:
                            log.error("[JS %s] All retries failed: %s", label, exc)
                            results = [{"error": True, "ratings": []}] * len(batch_tasks)
                        else:
                            await asyncio.sleep(1)

                # Process results — check for empty to trigger early stop
                batch_got_data = False
                for result in results:
                    if result.get("error"):
                        continue
                    ratings = result.get("ratings") or []
                    if not ratings:
                        consecutive_empty += 1
                        continue

                    batch_got_data = True
                    consecutive_empty = 0

                    for r in ratings:
                        rid = str(r.get("cmtid", ""))
                        if rid and rid in seen_ids:
                            continue
                        if rid:
                            seen_ids.add(rid)
                        rev = _normalize(r, url, product_name)
                        if rev:
                            async with buffer_lock:
                                buffer.append(rev)
                                count += 1

                                if len(buffer) >= BUFFER_SIZE:
                                    saved = exporter.save_batch(list(buffer))
                                    total_saved += saved
                                    buffer.clear()

                # Early stop: if 3+ consecutive requests return empty, star is done
                if consecutive_empty >= 3 or not batch_got_data:
                    break

            async with buffer_lock:
                stars_done.add(label)
                elapsed = time.time() - t0
                speed = count / elapsed if elapsed > 0 else 0
                pct = count / target * 100 if target > 0 else 0
                print(f"  [{label}] done | {count:,}/{target:,} ({pct:.1f}%) | {speed:.1f} rev/s | workers {len(stars_done)}/{n_workers}")

        # Run all 5 stars concurrently
        try:
            workers = [_fetch_star(s, f) for s in STAR_TYPES for f in self.filters]
            await asyncio.gather(*workers)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print(f"\n  Interrupt -- saving {count:,} reviews...")
        finally:
            # Flush remaining buffer
            async with buffer_lock:
                if buffer:
                    saved = exporter.save_batch(list(buffer))
                    total_saved += saved
                    buffer.clear()

            # Final checkpoint
            checkpoint.save({
                "item_id": item_id, "shop_id": shop_id,
                "total_on_server": total_count,
                "progress": progress, "reviews_count": total_saved,
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            })

            if total_saved >= target * 0.9:
                checkpoint.delete()
                print("  [Checkpoint] Complete -- deleted.")

            # Close browser
            try:
                await context.close()
            except Exception:
                pass

        elapsed = time.time() - t0
        speed = total_saved / max(elapsed, 1)
        print(f"\n  [{self.SITE_NAME}] Done: {total_saved:,}/{total_count:,} reviews"
              f" in {elapsed:.0f}s ({speed:.1f} rev/s)")
        return total_saved

    # ── Phase 1: CloakBrowser warm-up (keeps browser open) ────────────────

    async def _browser_warmup(self, url: str):
        """
        Returns (product_name, total_count, star_counts, page, context).
        Browser is kept OPEN for Phase 2.
        Returns ("", 0, {}, None, None) on failure.
        """
        from scraper.stealth_browser import launch_stealth_context

        product_name: str = ""
        total_count:  int = 0
        star_counts:  dict = {}
        got_response = asyncio.Event()

        async def _on_response(response) -> None:
            nonlocal total_count, star_counts
            if _RATINGS_PATH not in response.url or response.status != 200:
                return
            try:
                raw_bytes = await response.body()
                data = json.loads(raw_bytes.decode("utf-8", errors="replace"))
                inner = data.get("data") or {}
                summary = inner.get("item_rating_summary") or {}
                total = int(summary.get("rating_total") or 0)
                rc = summary.get("rating_count") or []
                if len(rc) >= 6:
                    star_counts = {1: rc[1], 2: rc[2], 3: rc[3], 4: rc[4], 5: rc[5]}
                if total > total_count:
                    total_count = total
                got_response.set()
                print(f"  [Warmup] API intercepted: total={total}, stars={star_counts}")
            except Exception as exc:
                print(f"  [Warmup] Parse error: {exc}")

        try:
            launch_kwargs = {
                "headless": self.headless,
                "humanize": self.humanize,
                "human_preset": self.human_preset,
            }
            if self._state_file.exists():
                print(f"  [Warmup] Loading login session: {self._state_file.name}")
                launch_kwargs["storage_state"] = str(self._state_file)
            else:
                print(f"  [Warmup] No login session, using persistent profile")
                launch_kwargs["user_data_dir"] = str(PROFILE_DIR)

            context = await launch_stealth_context(**launch_kwargs)
            page = await context.new_page()
            page.on("response", _on_response)

            await page.goto(url, wait_until="domcontentloaded", timeout=35_000)
            await page.wait_for_timeout(1500)

            # Product name
            try:
                h1 = page.locator("h1").first
                if await h1.count() > 0:
                    product_name = (await h1.inner_text()).strip()
                if not product_name:
                    product_name = (await page.title()).split("|")[0].strip()
            except Exception:
                pass

            # Scroll to trigger review API
            for pct in (0.4, 0.7, 1.0):
                await page.evaluate(f"window.scrollTo(0, document.body.scrollHeight * {pct})")
                await page.wait_for_timeout(800)

            for sel in _REVIEW_SECTION:
                try:
                    el = page.locator(sel).first
                    if await el.count() > 0:
                        await el.scroll_into_view_if_needed(timeout=2000)
                        await page.wait_for_timeout(500)
                        break
                except Exception:
                    continue

            # Wait for API response
            try:
                await asyncio.wait_for(
                    asyncio.shield(asyncio.ensure_future(_wait_event(got_response))),
                    timeout=25.0,
                )
            except (asyncio.TimeoutError, TimeoutError):
                try:
                    await page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                    await page.wait_for_timeout(2000)
                    await asyncio.wait_for(
                        asyncio.shield(asyncio.ensure_future(_wait_event(got_response))),
                        timeout=10.0,
                    )
                except (asyncio.TimeoutError, TimeoutError):
                    pass

            if total_count > 0:
                # Try to get star counts via JS fetch if not from interception
                if not star_counts:
                    try:
                        shop_id, item_id = _parse_ids(url)
                        result = await page.evaluate(
                            _COUNT_JS, {"shopid": shop_id, "itemid": item_id}
                        )
                        if not result.get("error"):
                            rc = result.get("counts") or []
                            if len(rc) >= 6:
                                star_counts = {1: rc[1], 2: rc[2], 3: rc[3], 4: rc[4], 5: rc[5]}
                    except Exception:
                        pass

                # Save session state
                try:
                    await context.storage_state(path=str(self._state_file))
                except Exception:
                    pass

                # Remove listener to avoid noise during Phase 2
                page.remove_listener("response", _on_response)

                # KEEP browser open — return page and context for Phase 2
                return product_name, total_count, star_counts, page, context

            # No reviews found — close and return failure
            await context.close()
            return "", 0, {}, None, None

        except Exception as exc:
            log.error("[Warmup] Error: %s", exc, exc_info=True)
            print(f"  [Warmup] Error: {exc}")
            try:
                await context.close()
            except Exception:
                pass
            return "", 0, {}, None, None


async def _wait_event(event: asyncio.Event) -> None:
    while not event.is_set():
        await asyncio.sleep(0.05)
