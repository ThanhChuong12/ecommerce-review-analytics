"""
CSV / JSON exporter with incremental (append) writes.
Data is saved after every page so nothing is lost if the agent crashes.
"""

import csv
import json
import os
from pathlib import Path

from scraper.models import Review

CSV_COLUMNS = ['product_name', 'text', 'rating', 'date', 'image_urls', 'product_url', 'scraped_at']


class ReviewExporter:
	"""Handles incremental export of reviews to CSV or JSON."""

	def __init__(self, output_path: str, fmt: str = 'csv'):
		self.output_path = Path(output_path)
		self.fmt = fmt.lower()
		# Create parent directories if they don't exist
		self.output_path.parent.mkdir(parents=True, exist_ok=True)
		self._header_written = False

	# ------------------------------------------------------------------
	# Public API
	# ------------------------------------------------------------------

	def save_batch(self, reviews: list[Review]) -> int:
		"""
		Append a batch of reviews to the output file.
		Returns the number of reviews successfully written.
		"""
		if not reviews:
			return 0

		try:
			if self.fmt == 'csv':
				return self._append_csv(reviews)
			else:
				return self._append_json(reviews)
		except Exception as e:
			print(f'  ⚠️  Export error: {e}')
			return 0

	def total_written(self) -> int:
		"""Return how many rows are currently in the output file."""
		try:
			if self.fmt == 'csv':
				if not self.output_path.exists():
					return 0
				with open(self.output_path, encoding='utf-8-sig') as f:
					# Subtract 1 for header row
					return max(0, sum(1 for _ in f) - 1)
			else:
				if not self.output_path.exists():
					return 0
				with open(self.output_path, encoding='utf-8') as f:
					data = json.load(f)
				return len(data)
		except Exception:
			return 0

	# ------------------------------------------------------------------
	# Internal helpers
	# ------------------------------------------------------------------

	def _append_csv(self, reviews: list[Review]) -> int:
		"""Append reviews to CSV using Python's csv module (RFC 4180 compliant)."""
		mode = 'a' if self._header_written or self.output_path.exists() else 'w'

		with open(self.output_path, mode, newline='', encoding='utf-8-sig') as f:
			writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS, quoting=csv.QUOTE_ALL)

			# Write header only once (first batch of a fresh file)
			if not self._header_written and mode == 'w':
				writer.writeheader()
				self._header_written = True
			elif not self._header_written:
				# File existed before we started — mark header as already there
				self._header_written = True

			count = 0
			for review in reviews:
				try:
					writer.writerow(
						{
							'product_name': review.product_name,
							'text': review.text,
							'rating': review.rating,
							'date': review.date,
							# Join multiple image URLs with pipe so one cell stays readable
							'image_urls': '|'.join(review.image_urls),
							'product_url': review.product_url,
							'scraped_at': review.scraped_at,
						}
					)
					count += 1
				except Exception as e:
					print(f'  ⚠️  Skipping row due to write error: {e}')

		return count

	def _append_json(self, reviews: list[Review]) -> int:
		"""Append reviews to JSON array (reads existing file, extends, rewrites)."""
		existing: list[dict] = []

		if self.output_path.exists():
			try:
				with open(self.output_path, encoding='utf-8') as f:
					existing = json.load(f)
			except (json.JSONDecodeError, OSError):
				existing = []

		new_rows = [r.model_dump() for r in reviews]
		existing.extend(new_rows)

		with open(self.output_path, 'w', encoding='utf-8') as f:
			json.dump(existing, f, ensure_ascii=False, indent=2)

		return len(new_rows)
