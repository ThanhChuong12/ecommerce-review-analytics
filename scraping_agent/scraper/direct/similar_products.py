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

_THIS_DIR = Path(__file__).resolve().parent.parent.parent
SESSION_DIR = _THIS_DIR / "output" / "agent_sessions"

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

def _extract_tiki_recs(data: object, depth: int = 0) -> list[dict]:
    if depth > 5:
        return []
    if isinstance(data, list) and data and isinstance(data[0], dict):
        sample = data[0]
        # Tiki uses 'name', 'price', 'id', 'thumbnail_url'
        if "name" in sample and "id" in sample and "price" in sample:
            return data
    if isinstance(data, dict):
        for v in data.values():
            found = _extract_tiki_recs(v, depth + 1)
            if found:
                return found
    return []

class TikiSimilar:
    """Lấy sản phẩm tương tự từ Tiki thông qua API tìm kiếm, vì API recommendation đã bị ẩn/đổi format."""

    SITE = "tiki"

    async def fetch(self, url: str, limit: int = 5) -> list[SimilarProduct]:
        import httpx
        import urllib.parse
        import re
        
        results: list[SimilarProduct] = []
        
        if "search?q=" in url:
            query = url.split("search?q=")[1]
        else:
            m = re.search(r"tiki\.vn/([^/]+)-p\d+", url)
            if not m:
                log.warning(f"[Tiki] Không trích xuất được slug từ {url}")
                return []
            slug = m.group(1)
            query = slug.replace("-", " ")
        
        search_api = f"https://tiki.vn/api/v2/products?limit={limit}&q={urllib.parse.quote(query)}"
        
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(search_api, headers={"User-Agent": "Mozilla/5.0"})
                if resp.status_code == 200:
                    data = resp.json()
                    items = data.get("data", [])
                    for p in items[:limit]:
                        pid   = str(p.get("id") or "")
                        name  = str(p.get("name") or "")
                        if not name or not pid: continue
                        price = int(p.get("price") or 0)
                        rating = float(p.get("rating_average") or 0)
                        
                        sold_raw = p.get("quantity_sold") or 0
                        if isinstance(sold_raw, dict):
                            sold = int(sold_raw.get("value") or 0)
                        else:
                            sold = int(sold_raw)
                            
                        img   = str(p.get("thumbnail_url") or "")
                        purl  = f"https://tiki.vn/{p.get('url_key', '')}-p{pid}.html" if pid else url
                        results.append(SimilarProduct(
                            name=name, url=purl, price=price,
                            rating=rating, sold=sold,
                            image_url=img, source=self.SITE,
                        ))
                    if results:
                        log.info(f"[Tiki] Đã dùng Search API tìm được {len(results)} sản phẩm tương tự")
                        return results
        except Exception as exc:
            log.error(f"[Tiki] Lỗi fetch search API: {exc}")
            
        log.warning("[Tiki] Không lấy được sản phẩm tương tự qua search API")
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

    # Chỉ bắt API gợi ý của Product Detail Page, tránh các API gợi ý của Shop hay Giỏ hàng
    _PATHS = [
        "detail.getrecommend"
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
            
            # Bỏ qua mtop.relation vì đây là API "Just For You" cá nhân hóa (trả về bạt trùm xe máy)
            if "mtop.relation" in url_lower:
                return
                
            if not any(p in url_lower for p in self._PATHS):
                return
            try:
                body  = await response.text()
                data  = json.loads(body)
                items = _extract_lazada_recs(data)
                if items:
                    for item in items:
                        r = _normalize_lazada_rec(item, self.SITE)
                        if r:
                            results.append(r)
                            if len(results) >= limit:
                                break
                    if results:
                        found_event.set()
            except Exception:
                pass

        try:
            # Dùng stealth_browser — tương tự LazadaScraper._scrape_attempt()
            # Không dùng storage_state cho Similar Products vì cookie đăng nhập làm Lazada ẩn API recommend
            context = await launch_stealth_context(
                storage_state=None,
                headless=self.headless,
                humanize=False,
            )
            page = await context.new_page()
            page.on("response", _on_response)

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            except Exception:
                pass

            is_verify = any(kw in page.url.lower() for kw in ["verify", "captcha", "security", "login"])
            if is_verify and not self.headless:
                print("  [Similar] 🔒 Lazada bot-check detected! Vui lòng giải quyết captcha trong trình duyệt (tối đa 5 phút)...")
                for _ in range(60):
                    await page.wait_for_timeout(5000)
                    if not any(kw in page.url.lower() for kw in ["verify", "captcha", "security", "login"]):
                        print("  [Similar] ✅ Verification successful! Continuing...")
                        break

            # Progressive scroll to trigger lazy-loaded recommendations (Lazada's observer needs sequential scrolling)
            for _ in range(15):
                if found_event.is_set():
                    break
                try:
                    await page.evaluate("window.scrollBy(0, 1000)")
                    await page.wait_for_timeout(1000)
                except Exception:
                    pass

            try:
                await asyncio.wait_for(_await_event(found_event), timeout=15.0)
            except asyncio.TimeoutError:
                if not found_event.is_set():
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
    if isinstance(data, dict):
        # Ưu tiên lấy module v2v (Sản phẩm tương tự)
        data_block = data.get("data")
        if isinstance(data_block, dict):
            for mod in data_block.values():
                if isinstance(mod, dict):
                    rec_type = mod.get("recommendType", "")
                    if rec_type in ["v2v", "item2item", "similar"]:
                        return mod.get("products") or mod.get("items") or []
    
    return []


def _normalize_lazada_rec(item: dict, source: str) -> SimilarProduct | None:
    try:
        name  = str(item.get("name") or item.get("title") or "")
        price_raw = item.get("price") or item.get("priceShow") or 0
        price_str = re.sub(r"[^\d]", "", str(price_raw))
        price = int(price_str) if price_str else 0
        
        rating_raw = item.get("ratingScore") or item.get("rating") or 0
        if isinstance(rating_raw, dict):
            rating_raw = rating_raw.get("average") or rating_raw.get("score") or 0
        rating = float(rating_raw)
        
        sold_raw = item.get("itemSoldCntShow") or item.get("sold") or 0
        sold = int(re.sub(r"[^\d]", "", str(sold_raw)) or 0)
        img    = str(item.get("image") or item.get("mainImage") or "")
        if img.startswith("//"):
            img = "https:" + img
        raw_url = str(item.get("itemUrl") or item.get("url") or "")
        if not raw_url:
            item_id = item.get("itemId") or item.get("item_id")
            if item_id:
                raw_url = f"https://www.lazada.vn/products/i{item_id}.html"

        if raw_url:
            if raw_url.startswith("//"):
                raw_url = "https:" + raw_url
            elif raw_url.startswith("/"):
                raw_url = "https://www.lazada.vn" + raw_url
            elif not raw_url.startswith("http"):
                raw_url = "https://www.lazada.vn/" + raw_url
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
                items = _extract_shopee_recs(data)
                if items:
                    for item in items[:limit]:
                        r = _normalize_shopee_rec(item, self.SITE)
                        if r:
                            results.append(r)
                    found_event.set()
            except Exception as e:
                pass

        try:
            state_file = SESSION_DIR / "state_shopee.vn.json"
            
            context = await launch_stealth_context(
                storage_state=str(state_file) if state_file.exists() else None,
                headless=self.headless,
                humanize=False,
                locale="vi-VN",
            )
            page = await context.new_page()
            page.on("response", _on_response)

            try:
                await page.goto(url, wait_until="domcontentloaded", timeout=30_000)
            except Exception:
                pass

            # Progressive scroll to trigger lazy-loaded recommendations
            for _ in range(15):
                if found_event.is_set():
                    break
                try:
                    await page.evaluate("window.scrollBy(0, 1000)")
                    await page.wait_for_timeout(1000)
                except Exception:
                    pass

            try:
                await asyncio.wait_for(
                    _await_event(found_event), timeout=10.0
                )
            except asyncio.TimeoutError:
                if not found_event.is_set():
                    log.warning("[Shopee] Recommendation API không phản hồi")

            await context.close()
        except Exception as exc:
            log.error("[Shopee] Similar fetch error: %s", exc)

        return results[:limit]


def _extract_shopee_recs(data: object, depth: int = 0) -> list[dict]:
    if depth > 5:
        return []
    if isinstance(data, list) and data and isinstance(data[0], dict):
        keys = data[0].keys()
        has_id = "itemid" in keys or "item_id" in keys
        has_shop = "shopid" in keys or "shop_id" in keys
        if has_id and has_shop:
            return data
        
        # If it's a list but not the target, recurse into its elements
        for item in data:
            found = _extract_shopee_recs(item, depth + 1)
            if found:
                return found

    elif isinstance(data, dict):
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

        # Fallback: Dùng Tiki API để tìm sản phẩm tương tự dựa trên tên
        _, _, _, product_name = self._parse_product_meta(html)
        if product_name:
            log.info("[TGDD] Fallback to Tiki search API using name: %s", product_name)
            tiki_url = f"https://tiki.vn/search?q={product_name}"
            try:
                tiki_fetcher = TikiSimilar()
                tiki_results = await tiki_fetcher.fetch(tiki_url, limit=limit)
                if tiki_results:
                    # Update source to tgdd_fallback so UI knows
                    for r in tiki_results:
                        r.source = "tgdd"
                    return tiki_results
            except Exception as e:
                log.warning("[TGDD] Fallback failed: %s", e)

        log.warning("[TGDD] Không lấy được sản phẩm tương tự")
        return []
