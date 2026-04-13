"""
dispatcher.py — routes a product URL to the correct scraper engine.

Approach 1 (fast, no browser, no LLM):
  tiki.vn           -> TikiScraper   (calls Tiki internal API v2)
  thegioididong.com -> TGDDScraper   (calls webapi.thegioididong.com)

Approach 1b (Playwright, no LLM):
  lazada.vn         -> LazadaScraper (network interception via Playwright)

Approach 2 (browser-use Agent, works for ANY site):
  everything else   -> scrape_reviews() from scraper/agent.py
                       Uses browser_use.Agent + Controller directly
                       (no custom wrapper, no pilot, just the library as intended)
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Approach 1: Direct API scrapers (no browser, no LLM)
# ---------------------------------------------------------------------------

_DIRECT_SITES = {
	'tiki.vn': 'scraper.direct.tiki.TikiScraper',
	'thegioididong.com': 'scraper.direct.tgdd.TGDDScraper',
}


def _get_direct_scraper(url: str):
	"""Return the direct scraper class if URL matches a known site, else None."""
	import importlib

	for domain, cls_path in _DIRECT_SITES.items():
		if domain in url:
			mod_name, cls_name = cls_path.rsplit('.', 1)
			mod = importlib.import_module(mod_name)
			return getattr(mod, cls_name)
	return None


# ---------------------------------------------------------------------------
# Main dispatch function
# ---------------------------------------------------------------------------


async def scrape(
	url: str,
	output_path: str,
	fmt: str = 'csv',
	max_reviews: int = 3000,
	llm_provider: str = 'auto',
	headless: bool = False,
) -> int:
	"""
	Route URL to the correct scraper.
	Returns total number of reviews saved.
	"""
	ScraperClass = _get_direct_scraper(url)

	if ScraperClass is not None:
		# Approach 1: fast direct API, no browser, no LLM
		scraper = ScraperClass()
		return await scraper.run(url, output_path, fmt, max_reviews)

	# ---------------------------------------------------------------------------
	# Approach 1b: Lazada — Playwright network interception (no LLM, no agent)
	# ---------------------------------------------------------------------------
	if 'lazada.vn' in url:
		from scraper.direct.lazada import LazadaScraper
		scraper = LazadaScraper(headless=headless)
		return await scraper.run(url, output_path, fmt, max_reviews)

	# Approach 2: browser_use.Agent — works for any site
	# Uses scraper/agent.py which calls browser_use.Agent directly
	# (same library that powers the browser_use examples — stable, no custom wrappers)
	from scraper.agent import scrape_reviews

	return await scrape_reviews(
		url=url,
		output_path=output_path,
		fmt=fmt,
		max_reviews=max_reviews,
		llm_provider=llm_provider,
		headless=headless,
	)
