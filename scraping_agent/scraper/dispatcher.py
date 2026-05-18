"""
dispatcher.py — Routes product URL đến đúng scraper engine.

Lớp 1 — Direct API (nhanh, không cần browser, không cần LLM):
  tiki.vn           → TikiScraper   (Tiki internal API v2)
  thegioididong.com → TGDDScraper   (webapi.thegioididong.com)

Lớp 2 — Playwright network interception (browser, không LLM):
  lazada.vn         → LazadaScraper  (dynamic tokens, cần browser)
  shopee.vn         → ShopeeScraper  (session + anti-bot, cần browser)
  bất kỳ site nào   → GenericPlaywrightScraper (auto-detect review API)

Lớp 3 — LLM browser agent (chậm, tốn tiền, chỉ dùng khi bắt buộc):
  bất kỳ site nào   → scraper/agent.py (browser_use.Agent)

Quy trình cho site lạ (không thuộc Lớp 1 hoặc 2 đã biết):
  Thử GenericPlaywrightScraper trước → nếu thất bại → LLM agent
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Lớp 1: Direct API scrapers (no browser, no LLM)
# ---------------------------------------------------------------------------

_DIRECT_SITES = {
    "tiki.vn":            "scraper.direct.tiki.TikiScraper",
    "thegioididong.com":  "scraper.direct.tgdd.TGDDScraper",
}


def _get_direct_scraper(url: str):
    """Trả về direct scraper class nếu URL khớp site đã biết, ngược lại None."""
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
) -> int:
    """Route URL đến đúng scraper. Trả về số review đã lưu."""

    # ── Lớp 1: Direct API (nhanh nhất) ─────────────────────────────────
    ScraperClass = _get_direct_scraper(url)
    if ScraperClass is not None:
        scraper = ScraperClass()
        return await scraper.run(url, output_path, fmt, max_reviews)

    # ── Lớp 2a: Lazada — Playwright interception ─────────────────────────
    if "lazada.vn" in url:
        from scraper.direct.lazada import LazadaScraper
        scraper = LazadaScraper(headless=headless)
        return await scraper.run(url, output_path, fmt, max_reviews)

    # ── Lớp 2b: Shopee — Playwright interception ─────────────────────────
    if "shopee.vn" in url:
        from scraper.direct.shopee import ShopeeScraper
        scraper = ShopeeScraper(headless=headless)
        return await scraper.run(url, output_path, fmt, max_reviews)

    # ── Lớp 2c: Site lạ — Generic Playwright (tự detect API) ────────────
    # Ưu tiên thử trước khi tốn tiền LLM
    try:
        from scraper.direct.generic_playwright import GenericPlaywrightScraper
        generic = GenericPlaywrightScraper(headless=headless)
        count = await generic.run(url, output_path, fmt, max_reviews)
        if count > 0:
            return count
        # count == 0 → không detect được → fallthrough sang LLM
        print(
            "  [Dispatcher] Generic scraper trả về 0 reviews → "
            "thử LLM agent..."
        )
    except RuntimeError as exc:
        # GenericPlaywrightScraper raise RuntimeError khi không detect được API
        print(f"  [Dispatcher] Generic scraper thất bại: {exc}")
        print("  [Dispatcher] Chuyển sang LLM agent...")

    # ── Lớp 3: LLM browser agent — fallback cuối cùng ───────────────────
    from scraper.agent import scrape_reviews
    return await scrape_reviews(
        url=url,
        output_path=output_path,
        fmt=fmt,
        max_reviews=max_reviews,
        llm_provider=llm_provider,
        headless=headless,
    )
