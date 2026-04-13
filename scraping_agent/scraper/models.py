"""
Data models for the review scraper.
Uses Pydantic v2 for validation + type safety.
"""

from datetime import datetime
from pydantic import BaseModel, Field, field_validator


class Review(BaseModel):
	"""A single product review."""

	review_id: str = Field(default='', description='Native review ID from the platform (used for deduplication)')
	text: str = Field(default='', description='Full review text content')
	rating: int = Field(default=5, description='Star rating 1-5')
	date: str = Field(default='', description='Review date as string')
	image_urls: list[str] = Field(default_factory=list, description='List of image URLs attached to the review')
	product_url: str = Field(default='', description='Original product URL')
	scraped_at: str = Field(default='', description='ISO timestamp when scraped')

	@field_validator('rating', mode='before')
	@classmethod
	def validate_rating(cls, v):
		"""Coerce rating to int 1-5, default 5 if invalid."""
		try:
			v = int(float(str(v)))
		except (ValueError, TypeError):
			return 5
		return max(1, min(5, v))

	@field_validator('image_urls', mode='before')
	@classmethod
	def validate_image_urls(cls, v):
		"""Accept list or single string; return list of valid URL strings."""
		if not v:
			return []
		if isinstance(v, str):
			stripped = v.strip()
			return [stripped] if stripped else []
		return [str(url).strip() for url in v if url and str(url).strip()]

	@field_validator('text', mode='before')
	@classmethod
	def validate_text(cls, v):
		"""Ensure text is always a string."""
		return str(v) if v is not None else ''

	@field_validator('date', mode='before')
	@classmethod
	def validate_date(cls, v):
		"""Ensure date is always a string."""
		return str(v) if v is not None else ''

	def model_post_init(self, __context):
		"""Auto-fill scraped_at if not provided."""
		if not self.scraped_at:
			self.scraped_at = datetime.now().isoformat()
