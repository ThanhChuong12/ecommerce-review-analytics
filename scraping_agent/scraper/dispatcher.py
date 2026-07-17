"""
dispatcher.py — Routes product URL to the correct scraper engine.

Level 1 — Direct API (Tiki, TGDD)
Level 2 — Playwright network interception (Lazada, Shopee, Generic)
Level 3 — LLM browser agent (Fallback for other sites)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Level 1: Direct API scrapers (no browser, no LLM)
# ---------------------------------------------------------------------------

_DIRECT_SITES = {
    "tiki.vn":            "scraper.direct.tiki.TikiScraper",
    "thegioididong.com":  "scraper.direct.tgdd.TGDDScraper",
}


def _get_direct_scraper(url: str):
    """Return direct scraper class if URL matches a known site, otherwise None."""
    import importlib
    for domain, cls_path in _DIRECT_SITES.items():
        if domain in url:
            mod_name, cls_name = cls_path.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            return getattr(mod, cls_name)
    return None


# ---------------------------------------------------------------------------
# Main dispatch function
# ---------------------------------------------------------------------------

async def scrape(
    url: str,
    output_path: str,
    fmt: str          = "csv",
    max_reviews: int  = 3000,
    llm_provider: str = "auto",
    headless: bool    = False,
    filter_mode: str  = "max",
    progress_callback = None
) -> int:
    """Route URL to correct scraper. Returns saved review count."""

    # -- Level 1: Direct API (fastest)
    ScraperClass = _get_direct_scraper(url)
    if ScraperClass is not None:
        scraper = ScraperClass()
        return await scraper.run(url, output_path, fmt, max_reviews, progress_callback=progress_callback)

    # -- Level 2a: Lazada - Playwright interception
    if "lazada.vn" in url:
        from scraper.direct.lazada import LazadaScraper
        scraper = LazadaScraper(headless=headless)
        return await scraper.run(url, output_path, fmt, max_reviews, progress_callback=progress_callback)

    # -- Level 2b: Shopee - Parallel HTTPX API fetch
    if "shopee.vn" in url:
        from scraper.direct.shopee_fast import ShopeeParallelScraper
        scraper = ShopeeParallelScraper(
            concurrency  = 30,
            api_limit    = 59,
            headless     = headless,
            humanize     = headless,  # Disable humanize in non-headless mode to allow manual login
            human_preset = "careful",
            filter_mode  = filter_mode,
        )
        return await scraper.run(url, output_path, fmt, max_reviews, progress_callback=progress_callback)

    # -- Level 2c: Unknown sites - Generic Playwright (auto-detect API)
    try:
        from scraper.direct.generic_playwright import GenericPlaywrightScraper
        generic = GenericPlaywrightScraper(headless=headless)
        count = await generic.run(url, output_path, fmt, max_reviews, progress_callback=progress_callback)
        if count > 0:
            return count
        # Fallback to LLM agent if no reviews detected
        print("  [Dispatcher] Generic scraper returned 0 reviews. Falling back to LLM agent.")
    except RuntimeError as exc:
        print(f"  [Dispatcher] Generic scraper failed: {exc}")
        print("  [Dispatcher] Falling back to LLM agent...")

    # -- Level 3: LLM browser agent (final fallback)
    try:
        from scraper.agent import scrape_reviews
    except ModuleNotFoundError as e:
        print(f"\n  [Dispatcher] LLM Agent unavailable: {e}")
        print("  Site not recognized and LLM agent not available.")
        return 0

    return await scrape_reviews(
        url=url,
        output_path=output_path,
        fmt=fmt,
        max_reviews=max_reviews,
        llm_provider=llm_provider,
        headless=headless,
    )
