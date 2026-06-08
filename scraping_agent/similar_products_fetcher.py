"""
similar_products_fetcher.py — Entry point công khai để lấy sản phẩm tương tự.

CÁCH DÙNG:
    from scraping_agent.similar_products_fetcher import scrape_similar_products

    products = asyncio.run(scrape_similar_products(
        url="https://tiki.vn/dien-thoai-samsung-p12345678.html",
        limit=5
    ))
    for p in products:
        print(p.name, p.price, p.url)

Hỗ trợ: tiki.vn | lazada.vn | shopee.vn | thegioididong.com
Trả về: list[SimilarProduct]  (xem scraper/models.py)
"""

from __future__ import annotations

import asyncio
import logging
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # chỉ import SimilarProduct khi cần để tránh circular

log = logging.getLogger(__name__)

# Mapping domain → fetcher class path (lazy import)
_SITE_MAP: dict[str, str] = {
    "tiki.vn":              "scraper.direct.similar_products.TikiSimilar",
    "lazada.vn":            "scraper.direct.similar_products.LazadaSimilar",
    "shopee.vn":            "scraper.direct.similar_products.ShopeeSimilar",
    "thegioididong.com":    "scraper.direct.similar_products.TGDDSimilar",
}


class UnsupportedSiteError(ValueError):
    """URL không thuộc site được hỗ trợ."""


def _get_fetcher_class(url: str):
    for domain, cls_path in _SITE_MAP.items():
        if domain in url.lower():
            import importlib
            mod_name, cls_name = cls_path.rsplit(".", 1)
            mod = importlib.import_module(mod_name)
            return getattr(mod, cls_name)
    supported = ", ".join(_SITE_MAP.keys())
    raise UnsupportedSiteError(
        f"Site không được hỗ trợ: {url}\n"
        f"Các site hỗ trợ: {supported}"
    )


async def scrape_similar_products(
    url: str,
    limit: int       = 5,
    headless: bool   = True,
) -> list:
    """Lấy tối đa `limit` sản phẩm tương tự từ URL sản phẩm.

    Args:
        url:      URL sản phẩm gốc (tiki.vn / lazada.vn / shopee.vn).
        limit:    Số sản phẩm tương tự cần lấy (mặc định 5).
        headless: Chạy browser ẩn (mặc định True). Đặt False nếu cần giải captcha.

    Returns:
        list[SimilarProduct] — có thể rỗng nếu không lấy được.

    Raises:
        UnsupportedSiteError: Nếu site chưa được hỗ trợ.
    """
    FetcherClass = _get_fetcher_class(url)

    # TikiSimilar không cần headless (dùng httpx)
    if "Tiki" in FetcherClass.__name__:
        fetcher = FetcherClass()
    else:
        fetcher = FetcherClass(headless=headless)

    log.info("Fetching %d similar products for %s via %s", limit, url, FetcherClass.__name__)
    results = await fetcher.fetch(url, limit=limit)
    log.info("Got %d similar products", len(results))
    return results


# ---------------------------------------------------------------------------
# CLI quicktest
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import sys
    import json as _json

    if len(sys.argv) < 2:
        print("Usage: python similar_products_fetcher.py <url> [limit]")
        sys.exit(1)

    _url   = sys.argv[1]
    _limit = int(sys.argv[2]) if len(sys.argv) > 2 else 5

    products = asyncio.run(scrape_similar_products(_url, limit=_limit, headless=True))

    if not products:
        print("Không lấy được sản phẩm tương tự.")
    else:
        print(f"\n=== {len(products)} Sản phẩm tương tự ===")
        for i, p in enumerate(products, 1):
            print(f"\n[{i}] {p.name}")
            print(f"    Giá   : {p.price:,} VND")
            print(f"    Rating: {p.rating:.1f} ⭐")
            print(f"    Đã bán: {p.sold:,}")
            print(f"    URL   : {p.url}")
