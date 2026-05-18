"""
similar_products.py — Scraper lấy "Sản phẩm tương tự" từ Tiki / Lazada / Shopee.

Mỗi class trả về list[SimilarProduct] với limit tối đa người dùng chỉ định.

Tiki:   Direct API (httpx) → nhanh, không cần browser
Lazada: Playwright network interception → bắt recommendation API
Shopee: Playwright network interception → bắt recommendation API
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import uuid
from pathlib import Path
from urllib.parse import urlparse

import httpx

from scraper.models import SimilarProduct

log = logging.getLogger("SimilarProducts")

SESSION_DIR = Path("output") / "agent_sessions"

_TIKI_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "application/json",
    "Accept-Language": "vi-VN,vi;q=0.9",
    "Referer": "https://tiki.vn/",
    "x-guest-token": uuid.uuid4().hex,
}


# ---------------------------------------------------------------------------
# Tiki — Direct API
# ---------------------------------------------------------------------------

class TikiSimilar:
    """Lấy sản phẩm tương tự từ Tiki qua internal API (httpx, nhanh)."""

    SITE = "tiki"

    # Thử nhiều endpoint theo thứ tự ưu tiên
    _ENDPOINTS = [
        "https://tiki.vn/api/v2/products/{id}/related",
        "https://tiki.vn/api/v2/widgets/parent-category-recommendations?product_id={id}",
        "https://tiki.vn/api/v2/recommendations?product_id={id}",
    ]

    def _parse_id(self, url: str) -> str:
        m = re.search(r"-p(\d+)(?:\.html)?(?:[?#]|$)", url)
        if m:
            return m.group(1)
        raise ValueError(f"Không tìm được product ID từ Tiki URL: {url}")

    async def fetch(self, url: str, limit: int = 5) -> list[SimilarProduct]:
        product_id = self._parse_id(url)
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            for endpoint_tpl in self._ENDPOINTS:
                endpoint = endpoint_tpl.format(id=product_id)
                try:
                    resp = await client.get(endpoint, headers=_TIKI_HEADERS)
                    if resp.status_code != 200:
                        continue
                    data  = resp.json()
                    items = (
                        data.get("data")
                        or data.get("products")
                        or data.get("items")
                        or []
                    )
                    if not items:
                        continue
                    results: list[SimilarProduct] = []
                    for p in items[:limit]:
                        pid   = str(p.get("id") or "")
                        name  = str(p.get("name") or p.get("product_name") or "")
                        price = int(p.get("price") or p.get("price_vnd") or 0)
                        rating = float(p.get("rating_average") or p.get("rating") or 0)
                        sold  = int(p.get("quantity_sold") or p.get("sold_quantity") or 0)
                        img   = str(
                            p.get("thumbnail_url")
                            or p.get("image")
                            or p.get("images", [""])[0]
                            or ""
                        )
                        purl  = f"https://tiki.vn/{p.get('url_key', '')}-p{pid}.html" if pid else url
                        results.append(SimilarProduct(
                            name=name, url=purl, price=price,
                            rating=rating, sold=sold,
                            image_url=img, source=self.SITE,
                        ))
                    if results:
                        log.info("[Tiki] %d similar products found", len(results))
                        return results
                except Exception as exc:
                    log.debug("[Tiki] Endpoint %s failed: %s", endpoint, exc)
                    continue
        log.warning("[Tiki] Không lấy được sản phẩm tương tự")
        return []


# ---------------------------------------------------------------------------
# Lazada — Playwright intercept
# ---------------------------------------------------------------------------

class LazadaSimilar:
    """Lấy sản phẩm tương tự từ Lazada qua Playwright network interception."""

    SITE = "lazada"

    # Paths API recommendation của Lazada
    _PATHS = [
        "api/v1/recommend",
        "mtop.alicom.wireless.recommend",
        "recommendation",
        "related_items",
    ]

    def __init__(self, headless: bool = True) -> None:
        self.headless = headless

    async def fetch(self, url: str, limit: int = 5) -> list[SimilarProduct]:
        from playwright.async_api import async_playwright

        results: list[SimilarProduct] = []
        found_event = asyncio.Event()

        async def _on_response(response) -> None:
            if response.status != 200:
                return
            url_lower = response.url.lower()
            if not any(p in url_lower for p in self._PATHS):
                return
            try:
                body  = await response.text()
                data  = json.loads(body)
                items = _extract_lazada_recs(data)
                if items:
                    for item in items[:limit]:
                        r = _normalize_lazada_rec(item, self.SITE)
                        if r:
                            results.append(r)
                    found_event.set()
            except Exception:
                pass

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
                    headless=self.headless,
                    args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
                )
                state_file = SESSION_DIR / "state_lazada.vn.json"
                ctx_kw: dict = {
                    "user_agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "locale": "vi-VN",
                }
                if state_file.exists():
                    ctx_kw["storage_state"] = str(state_file)

                ctx  = await browser.new_context(**ctx_kw)
                page = await ctx.new_page()
                page.on("response", _on_response)

                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.6)")
                await page.wait_for_timeout(3000)

                try:
                    await asyncio.wait_for(
                        _await_event(found_event), timeout=15.0
                    )
                except asyncio.TimeoutError:
                    log.warning("[Lazada] Recommendation API không phản hồi")

                await browser.close()
        except Exception as exc:
            log.error("[Lazada] Similar fetch error: %s", exc)

        return results[:limit]


def _extract_lazada_recs(data: object, depth: int = 0) -> list[dict]:
    if depth > 5:
        return []
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if "name" in data[0] or "title" in data[0] or "itemUrl" in data[0]:
            return data
    if isinstance(data, dict):
        for v in data.values():
            found = _extract_lazada_recs(v, depth + 1)
            if found:
                return found
    return []


def _normalize_lazada_rec(item: dict, source: str) -> SimilarProduct | None:
    try:
        name  = str(item.get("name") or item.get("title") or "")
        price_raw = item.get("price") or item.get("priceShow") or 0
        price = int(re.sub(r"[^\d]", "", str(price_raw)) or 0)
        rating = float(item.get("ratingScore") or item.get("rating") or 0)
        sold   = int(item.get("itemSoldCntShow") or item.get("sold") or 0)
        img    = str(item.get("image") or item.get("mainImage") or "")
        if img.startswith("//"):
            img = "https:" + img
        raw_url = str(item.get("itemUrl") or item.get("url") or "")
        if raw_url and not raw_url.startswith("http"):
            raw_url = "https://www.lazada.vn" + raw_url
        return SimilarProduct(
            name=name, url=raw_url, price=price,
            rating=rating, sold=sold, image_url=img, source=source,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shopee — Playwright intercept
# ---------------------------------------------------------------------------

class ShopeeSimilar:
    """Lấy sản phẩm tương tự từ Shopee qua Playwright network interception."""

    SITE = "shopee"

    _PATHS = [
        "api/v4/recommend",
        "api/v4/pdp/get_pc",
        "api/v4/item/get_related",
        "api/v4/search/search_items",
    ]

    def __init__(self, headless: bool = True) -> None:
        self.headless = headless

    async def fetch(self, url: str, limit: int = 5) -> list[SimilarProduct]:
        from playwright.async_api import async_playwright

        results: list[SimilarProduct] = []
        found_event = asyncio.Event()

        async def _on_response(response) -> None:
            if response.status != 200:
                return
            url_lower = response.url.lower()
            if not any(p in url_lower for p in self._PATHS):
                return
            try:
                body  = await response.text()
                data  = json.loads(body)
                items = _extract_shopee_recs(data)
                if items:
                    for item in items[:limit]:
                        r = _normalize_shopee_rec(item, self.SITE)
                        if r:
                            results.append(r)
                    found_event.set()
            except Exception:
                pass

        try:
            async with async_playwright() as pw:
                browser = await pw.chromium.launch(
                    headless=self.headless,
                    args=["--disable-blink-features=AutomationControlled", "--no-sandbox"],
                )
                state_file = SESSION_DIR / "state_shopee.vn.json"
                ctx_kw: dict = {
                    "user_agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                        "AppleWebKit/537.36 Chrome/124.0.0.0 Safari/537.36"
                    ),
                    "locale": "vi-VN",
                }
                if state_file.exists():
                    ctx_kw["storage_state"] = str(state_file)

                ctx  = await browser.new_context(**ctx_kw)
                page = await ctx.new_page()
                page.on("response", _on_response)

                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
                await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.5)")
                await page.wait_for_timeout(3000)

                try:
                    await asyncio.wait_for(
                        _await_event(found_event), timeout=15.0
                    )
                except asyncio.TimeoutError:
                    log.warning("[Shopee] Recommendation API không phản hồi")

                await browser.close()
        except Exception as exc:
            log.error("[Shopee] Similar fetch error: %s", exc)

        return results[:limit]


def _extract_shopee_recs(data: object, depth: int = 0) -> list[dict]:
    if depth > 5:
        return []
    if isinstance(data, list) and data and isinstance(data[0], dict):
        if any(k in data[0] for k in ("itemid", "item_id", "name", "shopid")):
            return data
    if isinstance(data, dict):
        for v in data.values():
            found = _extract_shopee_recs(v, depth + 1)
            if found:
                return found
    return []


def _normalize_shopee_rec(item: dict, source: str) -> SimilarProduct | None:
    try:
        name    = str(item.get("name") or item.get("item_name") or "")
        price_raw = item.get("price") or item.get("min_price") or 0
        # Shopee giá thường x100000
        price = int(price_raw)
        if price > 10_000_000_000:
            price //= 100_000
        rating  = float(item.get("item_rating", {}).get("rating_star", 0) if isinstance(item.get("item_rating"), dict) else 0)
        sold    = int(item.get("sold") or item.get("historical_sold") or 0)

        # Image URL Shopee
        images = item.get("images") or []
        img_id = images[0] if images else (item.get("image") or "")
        img    = f"https://cf.shopee.vn/file/{img_id}" if img_id and "/" not in img_id else str(img_id)

        shop_id = item.get("shopid") or item.get("shop_id") or ""
        item_id = item.get("itemid") or item.get("item_id") or ""
        purl    = f"https://shopee.vn/product/{shop_id}/{item_id}" if shop_id and item_id else ""

        return SimilarProduct(
            name=name, url=purl, price=price,
            rating=rating, sold=sold, image_url=img, source=source,
        )
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Shared util
# ---------------------------------------------------------------------------

async def _await_event(event: asyncio.Event) -> None:
    while not event.is_set():
        await asyncio.sleep(0.05)
    event.clear()
