"""
BaseScraper — shared template for all direct API scrapers.

Subclasses implement:
  _parse_product_id(url) -> str
  _get_total_pages(client, product_id) -> int | None
  _fetch_page(client, product_id, page, url) -> list[Review]

The base class handles: retry logic, deduplication, incremental saving, rate limiting.
"""

import asyncio
import hashlib
from abc import ABC, abstractmethod

import httpx

from scraper.exporter import ReviewExporter
from scraper.models import Review


class BaseScraper(ABC):
	# Subclasses configure these
	SITE_NAME: str = 'unknown'
	PAGE_SIZE: int = 20
	DELAY_SEC: float = 0.5
	MAX_RETRIES: int = 3

	def __init__(self) -> None:
		self._seen: set[str] = set()  # dedup: hash → already saved

	# ------------------------------------------------------------------
	# Public entry point
	# ------------------------------------------------------------------

	async def run(
		self,
		url: str,
		output_path: str,
		fmt: str = 'csv',
		max_reviews: int = 3000,
	) -> int:
		"""Scrape reviews and write to output_path. Returns count saved."""
		print(f'  [{self.SITE_NAME}] Direct API scraper — no browser / no LLM needed')

		try:
			product_id = self._parse_product_id(url)
		except Exception as exc:
			raise ValueError(f'Cannot parse product ID from URL: {exc}') from exc

		print(f'  Product ID: {product_id}')
		exporter = ReviewExporter(output_path, fmt)
		total_saved = 0
		page = 1
		total_pages: int | None = None

		async with httpx.AsyncClient(timeout=30, follow_redirects=True) as client:
			# Fetch total pages on first request
			total_pages = await self._get_total_pages(client, product_id)
			if total_pages:
				print(f'  Total pages: {total_pages}')

			while total_saved < max_reviews:
				if total_pages and page > total_pages:
					print('  All pages scraped.')
					break

				# Retry logic
				reviews: list[Review] = []
				for attempt in range(1, self.MAX_RETRIES + 1):
					try:
						reviews = await self._fetch_page(client, product_id, page, url)
						break
					except Exception as exc:
						print(f'  Page {page}, attempt {attempt}/{self.MAX_RETRIES}: {exc}')
						if attempt < self.MAX_RETRIES:
							await asyncio.sleep(2 ** attempt)
						else:
							print(f'  Skipping page {page} after {self.MAX_RETRIES} failures.')

				if not reviews:
					print(f'  No reviews on page {page} — stopping.')
					break

				# Deduplication: ưu tiên dùng native review_id (ổn định, chính xác)
				# Fallback sang hash nội dung nếu scraper không cung cấp ID
				new_reviews: list[Review] = []
				for r in reviews:
					if r.review_id:
						key = f'id:{r.review_id}'
					else:
						key = hashlib.md5(
							f'{r.text[:120]}|{r.date}|{r.rating}'.encode()
						).hexdigest()
					if key not in self._seen:
						self._seen.add(key)
						new_reviews.append(r)

				saved = exporter.save_batch(new_reviews)
				total_saved += saved
				skipped = len(reviews) - len(new_reviews)
				print(
					f'  [OK] Page {page}: +{saved} reviews'
					+ (f' ({skipped} dupes skipped)' if skipped else '')
					+ f' — Total: {total_saved:,}/{max_reviews:,}'
				)

				if total_saved >= max_reviews:
					print(f'  Target {max_reviews:,} reached.')
					break

				page += 1
				await asyncio.sleep(self.DELAY_SEC)

		return total_saved

	# ------------------------------------------------------------------
	# Abstract interface — override in each site scraper
	# ------------------------------------------------------------------

	@abstractmethod
	def _parse_product_id(self, url: str) -> str:
		"""Extract the product identifier (ID, slug, etc.) from the URL."""
		...

	@abstractmethod
	async def _get_total_pages(
		self, client: httpx.AsyncClient, product_id: str
	) -> int | None:
		"""Return the total number of review pages (None if unknown)."""
		...

	@abstractmethod
	async def _fetch_page(
		self,
		client: httpx.AsyncClient,
		product_id: str,
		page: int,
		product_url: str,
	) -> list[Review]:
		"""Fetch one page of reviews and return parsed Review objects."""
		...
