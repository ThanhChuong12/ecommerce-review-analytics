"""
similar_products.py — Scraper lấy "Sản phẩm tương tự" từ Tiki / Lazada / Shopee / TGDD.

Mỗi class trả về list[SimilarProduct] với limit tối đa người dùng chỉ định.

Tiki:   Direct API (httpx) → nhanh, không cần browser
Lazada: stealth_browser (CloakBrowser→Playwright) + persistent session → tránh captcha
Shopee: Playwright network interception → bắt recommendation API
TGDD:   Direct API (httpx) → parse từ webapi.thegioididong.com
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
# Lazada — stealth_browser intercept (CloakBrowser → Playwright fallback)
# ---------------------------------------------------------------------------

class LazadaSimilar:
    """Lấy sản phẩm tương tự từ Lazada qua stealth_browser network interception.

    Dùng launch_stealth_context (cùng pattern với LazadaScraper) để:
    - Tái sử dụng session state từ state_lazada.vn.json (tránh re-captcha)
    - CloakBrowser → Playwright fallback tự động
    """

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
        SESSION_DIR.mkdir(parents=True, exist_ok=True)
        self._state_file = SESSION_DIR / "state_lazada.vn.json"

    async def fetch(self, url: str, limit: int = 5) -> list[SimilarProduct]:
        from scraper.stealth_browser import launch_stealth_context

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
            # Dùng stealth_browser — tương tự LazadaScraper._scrape_attempt()
            # Chia sẻ session state với LazadaScraper → tránh re-trigger captcha
            context = await launch_stealth_context(
                storage_state=str(self._state_file) if self._state_file.exists() else None,
                headless=self.headless,
                humanize=False,
            )
            page = await context.new_page()
            page.on("response", _on_response)

            await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            await page.evaluate("window.scrollTo(0, document.body.scrollHeight * 0.6)")
            await page.wait_for_timeout(3000)

            try:
                await asyncio.wait_for(_await_event(found_event), timeout=15.0)
            except asyncio.TimeoutError:
                log.warning("[Lazada] Recommendation API không phản hồi")

            # Lưu session state sau mỗi lần chạy thành công
            try:
                await context.storage_state(path=str(self._state_file))
            except Exception:
                pass

            await context.close()
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


# ---------------------------------------------------------------------------
# TGDD — Direct API (httpx, không cần browser)
# ---------------------------------------------------------------------------

_TGDD_BASE = "https://www.thegioididong.com"
_TGDD_WEBAPI = "https://webapi.thegioididong.com"

_TGDD_HEADERS_HTML = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "vi-VN,vi;q=0.9",
    "Referer": _TGDD_BASE + "/",
}


class TGDDSimilar:
    """Lấy sản phẩm tương tự từ TGDD qua Direct API (httpx, không cần browser).

    Flow:
      1. GET trang sản phẩm → parse data-objectid
      2. GET /json/product/related?objectid={id} → list sản phẩm liên quan
    """

    SITE = "tgdd"

    # Các endpoint thử theo thứ tự ưu tiên
    _RELATED_ENDPOINTS = [
        "{base}/json/product/related?objectid={id}&objecttype={type}&siteid={site}&pagesize={limit}",
        "{base}/json/product/recommendproducts?objectid={id}&pagesize={limit}",
    ]

    def _parse_product_meta(self, html: str) -> tuple[str, str, str, str]:
        """Trích (object_id, object_type, site_id, product_name) từ HTML trang sản phẩm."""
        import re as _re
        from html import unescape

        oid   = _re.search(r'data-objectid="(\d+)"', html)
        otype = _re.search(r'data-objecttype="(\d+)"', html)
        sid   = _re.search(r'data-siteid="(\d+)"', html)
        h1    = _re.search(r'<h1[^>]*>\s*(.*?)\s*</h1>', html, _re.IGNORECASE | _re.DOTALL)

        object_id   = oid.group(1)   if oid   else ""
        object_type = otype.group(1) if otype else "2"
        site_id     = sid.group(1)   if sid   else "1"
        name = unescape(_re.sub(r'<[^>]+>', '', h1.group(1)).strip()) if h1 else ""
        return object_id, object_type, site_id, name

    def _normalize_tgdd_product(self, item: dict) -> SimilarProduct | None:
        """Chuẩn hóa 1 item từ TGDD related API → SimilarProduct."""
        try:
            name     = str(item.get("Name") or item.get("ProductName") or "")
            price    = int(item.get("Price") or item.get("FinalPrice") or 0)
            rating   = float(item.get("RatingStar") or item.get("Star") or 0)
            sold     = int(item.get("SoldQuantity") or item.get("TotalSold") or 0)
            img      = str(item.get("Image") or item.get("Thumbnail") or "")
            if img and not img.startswith("http"):
                img = "https://" + img.lstrip("/")
            slug     = str(item.get("Url") or item.get("ProductUrl") or "")
            purl     = f"{_TGDD_BASE}{slug}" if slug.startswith("/") else slug
            if not name:
                return None
            return SimilarProduct(
                name=name, url=purl, price=price,
                rating=rating, sold=sold, image_url=img, source=self.SITE,
            )
        except Exception:
            return None

    async def fetch(self, url: str, limit: int = 5) -> list[SimilarProduct]:
        import re as _re

        results: list[SimilarProduct] = []
        async with httpx.AsyncClient(timeout=15, follow_redirects=True) as client:
            # Bước 1: Lấy metadata từ trang sản phẩm
            try:
                resp = await client.get(url, headers=_TGDD_HEADERS_HTML)
                resp.raise_for_status()
                html = resp.text
            except Exception as exc:
                log.warning("[TGDD] Cannot fetch product page: %s", exc)
                return []

            object_id, object_type, site_id, _ = self._parse_product_meta(html)
            if not object_id:
                log.warning("[TGDD] Cannot parse data-objectid from: %s", url)
                return []

            log.info("[TGDD] objectId=%s | type=%s | site=%s", object_id, object_type, site_id)

            # Bước 2: Gọi related products API
            for endpoint_tpl in self._RELATED_ENDPOINTS:
                endpoint = endpoint_tpl.format(
                    base=_TGDD_WEBAPI,
                    id=object_id,
                    type=object_type,
                    site=site_id,
                    limit=limit,
                )
                try:
                    resp = await client.get(endpoint, headers=_TGDD_HEADERS_HTML)
                    if resp.status_code != 200:
                        continue
                    data = resp.json()
                    # TGDD thường trả về list trực tiếp hoặc trong key "Data"/"Products"
                    items = (
                        data if isinstance(data, list)
                        else data.get("Data") or data.get("Products") or []
                    )
                    if not items:
                        continue
                    for item in items[:limit]:
                        p = self._normalize_tgdd_product(item)
                        if p:
                            results.append(p)
                    if results:
                        log.info("[TGDD] %d similar products found", len(results))
                        return results
                except Exception as exc:
                    log.debug("[TGDD] Endpoint %s failed: %s", endpoint, exc)
                    continue

        log.warning("[TGDD] Không lấy được sản phẩm tương tự")
        return []
