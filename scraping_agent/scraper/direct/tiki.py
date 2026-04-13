"""
TikiScraper — calls Tiki's internal review API directly.

API:  https://tiki.vn/api/v2/reviews?product_id={id}&page={n}&limit=20
Auth: x-guest-token = any random UUID (no login required)
"""

import re
import uuid
from datetime import datetime

import httpx

from scraper.direct.base import BaseScraper
from scraper.models import Review

_HEADERS = {
	'User-Agent': (
		'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
		'AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36'
	),
	'Accept': 'application/json',
	'Accept-Language': 'vi-VN,vi;q=0.9,en-US;q=0.8',
	'Referer': 'https://tiki.vn/',
	'Origin': 'https://tiki.vn',
	'x-guest-token': uuid.uuid4().hex,  # Tiki requires this header (any UUID)
}

_API_URL = 'https://tiki.vn/api/v2/reviews'


class TikiScraper(BaseScraper):
	SITE_NAME = 'Tiki'
	PAGE_SIZE = 20
	DELAY_SEC = 0.4

	# ------------------------------------------------------------------
	# URL parser
	# ------------------------------------------------------------------

	def _parse_product_id(self, url: str) -> str:
		"""Extract numeric product ID from Tiki URL.

		Patterns:
		  tiki.vn/product-name-p277596856.html
		  tiki.vn/product-name-p277596856
		"""
		m = re.search(r'-p(\d+)(?:\.html)?(?:[?#]|$)', url)
		if m:
			return m.group(1)
		raise ValueError(
			f'No product ID found in Tiki URL: {url}\n'
			'Expected pattern: tiki.vn/product-name-p<ID>.html'
		)

	# ------------------------------------------------------------------
	# Total pages
	# ------------------------------------------------------------------

	async def _get_total_pages(
		self, client: httpx.AsyncClient, product_id: str
	) -> int | None:
		try:
			resp = await client.get(
				_API_URL,
				params={'product_id': product_id, 'page': 1, 'limit': self.PAGE_SIZE},
				headers=_HEADERS,
			)
			resp.raise_for_status()
			data = resp.json()
			total = data.get('paginationInfo', {}).get('totalPage')
			return int(total) if total else None
		except Exception:
			return None  # unknown, will stop when pages return empty

	# ------------------------------------------------------------------
	# Fetch one page
	# ------------------------------------------------------------------

	async def _fetch_page(
		self,
		client: httpx.AsyncClient,
		product_id: str,
		page: int,
		product_url: str,
	) -> list[Review]:
		resp = await client.get(
			_API_URL,
			params={
				'product_id': product_id,
				'page': page,
				'limit': self.PAGE_SIZE,
				# sort by id|desc để đảm bảo thứ tự ổn định giữa các trang (không bị overlap)
				'sort': 'id|desc',
			},
			headers=_HEADERS,
		)
		resp.raise_for_status()
		data = resp.json()
		reviews: list[Review] = []

		for item in data.get('data', []):
			try:
				# Native review ID — dùng làm dedup key chính xác hơn hash nội dung
				review_id = str(item.get('id') or '')

				# Image URLs — Tiki stores full_path or url
				raw_imgs = item.get('images') or []
				img_urls = [
					img.get('full_path') or img.get('url', '')
					for img in raw_imgs
					if isinstance(img, dict)
				]
				img_urls = [u for u in img_urls if u]

				# Date — Tiki gives Unix timestamp
				ts = item.get('created_at')
				date_str = (
					datetime.fromtimestamp(int(ts)).strftime('%Y-%m-%d') if ts else ''
				)

				# Text — "body" is the main review field
				text = ' '.join(str(item.get('body') or item.get('content') or '').split())

				reviews.append(
					Review(
						review_id=review_id,
						text=text,
						rating=int(item.get('rating', 5)),
						date=date_str,
						image_urls=img_urls,
						product_url=product_url,
						scraped_at=datetime.now().isoformat(),
					)
				)
			except Exception as exc:
				# Skip individual malformed reviews, don't crash
				print(f'    Skip malformed review: {exc}')

		return reviews
