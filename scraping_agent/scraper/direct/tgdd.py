"""
TGDDScraper — học từ code chuẩn, dùng 2 bước:

  Bước 1: GET trang sản phẩm → parse data-objectid / data-objecttype / data-siteid
  Bước 2: POST webapi.thegioididong.com/comment/init → nhận HTML reviews trang 1
           POST webapi.thegioididong.com/comment/list → trang 2, 3, ...

URL sản phẩm chuẩn:
  https://www.thegioididong.com/dtdd/samsung-galaxy-a06-5g-6gb-128gb
  https://www.thegioididong.com/dtdd/samsung-galaxy-a06-5g-6gb-128gb/danh-gia  ← tự strip

HTML pattern đã xác nhận:
  .cmt-top block → iconcmt-starbuy (count = rating), cmt-content (text), dd/MM/yyyy (date)
"""

import math
import re
from datetime import datetime
from urllib.parse import urlparse

import httpx

from scraper.direct.base import BaseScraper
from scraper.models import Review

_TGDD_BASE = 'https://www.thegioididong.com'
_WEBAPI    = 'https://webapi.thegioididong.com'

_HEADERS_HTML = {
	'User-Agent': (
		'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
		'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
	),
	'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
	'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8',
	'Referer': _TGDD_BASE + '/',
	'Origin': _TGDD_BASE,
}

_HEADERS_API = {
	**_HEADERS_HTML,
	'Accept': 'text/html, */*; q=0.01',
	'Content-Type': 'application/x-www-form-urlencoded; charset=UTF-8',
	'X-Requested-With': 'XMLHttpRequest',
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _normalize_url(url: str) -> str:
	"""Strip /danh-gia suffix và query params → slug URL chuẩn."""
	url = re.sub(r'/danh-gia/?$', '', url.rstrip('/'))
	parsed = urlparse(url)
	# Keep scheme + netloc + path only
	return f'{parsed.scheme}://{parsed.netloc}{parsed.path}'


def _extract_slug(url: str) -> str:
	"""Lấy phần slug từ URL TGDD.

	  /dtdd/samsung-galaxy-a06-5g-6gb-128gb  →  samsung-galaxy-a06-5g-6gb-128gb
	  /dt/iphone-16-pro-i-16612391          →  iphone-16-pro-i-16612391
	"""
	m = re.search(
		r'thegioididong\.com/(?:dtdd|dt|dien-thoai|laptop|tablet|may-tinh-bang)'
		r'/([^/?#\s]+)',
		url,
	)
	if m:
		return m.group(1)
	# Fallback: last path segment
	parts = urlparse(url).path.strip('/').split('/')
	if parts:
		return parts[-1]
	raise ValueError(f'Cannot extract slug from TGDD URL: {url}')


def _parse_cmt_blocks(html: str, product_url: str) -> list[Review]:
	"""Parse tất cả .cmt-top blocks trong HTML fragment."""
	blocks = re.split(r'(?=<div class="cmt-top")', html)
	reviews: list[Review] = []

	for block in blocks:
		if 'cmt-top-star' not in block:
			continue
		try:
			r = _parse_one_block(block, product_url)
			if r:
				reviews.append(r)
		except Exception:
			pass

	return reviews


def _parse_one_block(block: str, product_url: str) -> Review | None:
	# ----- Rating: đếm iconcmt-starbuy -----
	rating = len(re.findall(r'iconcmt-starbuy(?!")', block))
	if rating == 0:
		rating = len(re.findall(r'class="[^"]*starbuy[^"]*"', block))
	rating = max(1, min(5, rating)) if rating else 5

	# ----- Date: dd/MM/yyyy -----
	date_str = ''
	date_m = re.search(r'(\d{2}/\d{2}/\d{4})', block)
	if date_m:
		try:
			d, mo, y = date_m.group(1).split('/')
			date_str = f'{y}-{mo}-{d}'
		except Exception:
			date_str = date_m.group(1)

	# ----- Text: cmt-content div -----
	text = ''
	cm = re.search(r'class="cmt-content[^"]*">(.*?)</(?:div|p)>', block, re.DOTALL)
	if cm:
		text = re.sub(r'<[^>]+>', ' ', cm.group(1))
		text = re.sub(r'\s+', ' ', text).strip()

	# ----- Images -----
	imgs = re.findall(
		r'(?:src|href)="(https?://cdn(?:v2)?\.tgdd\.vn/[^"]+\.(?:jpg|jpeg|png|webp))"',
		block,
		re.IGNORECASE,
	)

	if not text and not date_str and rating == 0:
		return None

	return Review(
		text=text,
		rating=rating,
		date=date_str,
		image_urls=imgs,
		product_url=product_url,
		scraped_at=datetime.now().isoformat(),
	)


# ---------------------------------------------------------------------------
# Scraper class
# ---------------------------------------------------------------------------


class TGDDScraper(BaseScraper):
	SITE_NAME = 'TGDD'
	PAGE_SIZE = 10
	DELAY_SEC = 0.6

	def __init__(self) -> None:
		super().__init__()
		# Populated in _get_total_pages (which runs first)
		self._object_id: str = ''
		self._object_type: str = '2'
		self._site_id: str = '1'
		self._product_url_clean: str = ''
		self._total_reviews: int = 0

	# ------------------------------------------------------------------
	# URL → product ID (slug, used only as a key here)
	# ------------------------------------------------------------------

	def _parse_product_id(self, url: str) -> str:
		# Normalize — strip /danh-gia etc.
		self._product_url_clean = _normalize_url(url)
		return _extract_slug(self._product_url_clean)

	# ------------------------------------------------------------------
	# Step 1: GET product page → extract objectId/objectType/siteId
	# ------------------------------------------------------------------

	async def _get_total_pages(
		self, client: httpx.AsyncClient, product_id: str
	) -> int | None:
		# Build product page URL (handle /dt/ vs /dtdd/ automatically)
		if self._product_url_clean:
			product_url = self._product_url_clean
		else:
			product_url = f'{_TGDD_BASE}/dtdd/{product_id}'

		# Warm session
		try:
			await client.get(product_url, headers=_HEADERS_HTML)
		except Exception:
			pass

		resp = await client.get(product_url, headers=_HEADERS_HTML)
		resp.raise_for_status()
		html = resp.text

		oid   = re.search(r'data-objectid="(\d+)"',   html)
		otype = re.search(r'data-objecttype="(\d+)"', html)
		sid   = re.search(r'data-siteid="(\d+)"',     html)

		if not oid:
			raise ValueError(
				f'Cannot find data-objectid in page: {product_url}'
			)

		self._object_id   = oid.group(1)
		self._object_type = otype.group(1) if otype else '2'
		self._site_id     = sid.group(1)   if sid   else '1'

		print(f'  objectId={self._object_id} | type={self._object_type} | site={self._site_id}')

		# Tìm tổng số review từ nhiều nguồn
		total = self._extract_total_from_html(html)
		if total == 0:
			total = await self._fetch_total_from_danh_gia(client, product_id)

		self._total_reviews = total
		if total > 0:
			pages = max(1, math.ceil(total / self.PAGE_SIZE))
			print(f'  Total reviews detected: {total} -> {pages} pages')
			return pages

		return None  # không biết tổng, dừng khi trống

	def _extract_total_from_html(self, html: str) -> int:
		"""Trích tổng review từ HTML sản phẩm hoặc /danh-gia."""
		for pattern in [
			r'(\d+)\s*d.nh gi.',  # danh gia (broad)
			r'(\d+)\s*nh.n x.t',  # nhan xet
			r'"total"\s*:\s*(\d+)',
			r'rating_total[":\s]+(\d+)',
		]:
			m = re.search(pattern, html, re.IGNORECASE)
			if m:
				val = int(m.group(1))
				if 0 < val < 200_000:
					return val
		return 0

	async def _fetch_total_from_danh_gia(
		self, client: httpx.AsyncClient, product_id: str
	) -> int:
		"""Lấy tổng từ trang /danh-gia (có nhiều hint hơn)."""
		danh_gia_url = self._product_url_clean.rstrip('/') + '/danh-gia'
		try:
			r = await client.get(danh_gia_url, headers=_HEADERS_HTML)
			return self._extract_total_from_html(r.text)
		except Exception:
			return 0

	# ------------------------------------------------------------------
	# Step 2: POST comment API → HTML với .cmt-top blocks
	# ------------------------------------------------------------------

	async def _fetch_page(
		self,
		client: httpx.AsyncClient,
		product_id: str,
		page: int,
		product_url: str,
	) -> list[Review]:
		html = await self._post_comment_api(client, page)
		return _parse_cmt_blocks(html, self._product_url_clean or product_url)

	async def _post_comment_api(
		self, client: httpx.AsyncClient, page: int
	) -> str:
		"""
		Dùng /comment/allrating với pageIndex (0-based) cho tất cả các trang.
		Đây là endpoint xác nhận hoạt động: pageIndex=0 → trang 1, pageIndex=1 → trang 2, ...
		"""
		payload = {
			'objectId':   self._object_id,
			'objectType': self._object_type,
			'siteId':     self._site_id,
			'pageSize':   str(self.PAGE_SIZE),
			'pageIndex':  str(page - 1),  # 0-based
			'isstaging':  'false',
			'ismb':       'false',
			'url':        self._product_url_clean,
		}
		endpoint = f'{_WEBAPI}/comment/allrating'
		resp = await client.post(endpoint, data=payload, headers=_HEADERS_API)
		resp.raise_for_status()
		return resp.text
