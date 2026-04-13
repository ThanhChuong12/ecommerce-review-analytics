"""
E-commerce Review Scraper — CLI entry point.

Usage:
    python main.py "https://shopee.vn/..."
    python main.py "https://shopee.vn/..." --output reviews.csv --max-reviews 5000
    python main.py "https://shopee.vn/..." --format json --llm openai
    python main.py --help
"""

import argparse
import asyncio
import os
import sys
from datetime import datetime
from pathlib import Path

# Make sure we can import from the browser-use root and our scraper package
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Fix Windows cp1252 encoding — allow Unicode symbols in print statements
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')


from dotenv import load_dotenv, find_dotenv

# Tìm .env bắt đầu từ thư mục chứa file này, leo dần lên root repo
# → luôn hoạt động dù chạy lệnh từ scraping_agent/ hay từ final ML/
load_dotenv(find_dotenv(usecwd=False))


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		prog='python main.py',
		description='E-commerce Review Scraper powered by browser-use AI Agent',
		formatter_class=argparse.RawDescriptionHelpFormatter,
		epilog="""
Examples:
  python main.py "https://shopee.vn/product/123"
  python main.py "https://shopee.vn/product/123" --max-reviews 5000
  python main.py "https://shopee.vn/product/123" --output my_reviews.csv
  python main.py "https://shopee.vn/product/123" --format json
  python main.py "https://shopee.vn/product/123" --llm openai

Supported LLM providers (set the matching key in .env):
  browseruse  ->  BROWSER_USE_API_KEY   (fastest, recommended)
  openai      ->  OPENAI_API_KEY
  google      ->  GOOGLE_API_KEY
  groq        ->  GROQ_API_KEY          (free tier, rate-limited)
		""",
	)

	parser.add_argument(
		'url',
		help='Product URL to scrape reviews from (Shopee, Lazada, Tiki, etc.)',
	)
	parser.add_argument(
		'--output',
		'-o',
		default=None,
		metavar='FILE',
		help='Output file path. Default: reviews_YYYYMMDD_HHMMSS.csv',
	)
	parser.add_argument(
		'--format',
		'-f',
		choices=['csv', 'json'],
		default='csv',
		help='Output format (default: csv)',
	)
	parser.add_argument(
		'--max-reviews',
		'-n',
		type=int,
		default=3000,
		metavar='N',
		help='Stop after collecting N reviews (default: 3000)',
	)
	parser.add_argument(
		'--llm',
		choices=['auto', 'browseruse', 'openai', 'google', 'groq'],
		default='auto',
		metavar='PROVIDER',
		help='LLM provider: auto | browseruse | openai | google | groq (default: auto)',
	)
	parser.add_argument(
		'--headless',
		action='store_true',
		help='Run browser without a visible window (not recommended for Shopee)',
	)

	return parser.parse_args()


def print_banner(url: str, output: str, fmt: str, max_reviews: int, llm: str, headless: bool) -> None:
	bar = '=' * 60
	print(f'\n{bar}')
	print('  E-commerce Review Scraper')
	print(bar)
	print(f'  URL        : {url}')
	print(f'  Output     : {output}')
	print(f'  Format     : {fmt.upper()}')
	print(f'  Target     : {max_reviews:,} reviews')
	print(f'  LLM        : {llm}')
	print(f'  Headless   : {headless}')
	print(bar)
	print('  Press Ctrl+C at any time to stop -- progress is saved after every page.\n')


async def run(args: argparse.Namespace) -> int:
	from scraper.dispatcher import scrape

	return await scrape(
		url=args.url,
		output_path=args.output,
		fmt=args.format,
		max_reviews=args.max_reviews,
		llm_provider=args.llm,
		headless=args.headless,
	)


def main() -> None:
	args = parse_args()

	# Build default output filename
	if args.output is None:
		timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
		args.output = f'reviews_{timestamp}.{args.format}'

	print_banner(
		url=args.url,
		output=args.output,
		fmt=args.format,
		max_reviews=args.max_reviews,
		llm=args.llm,
		headless=args.headless,
	)

	try:
		total = asyncio.run(run(args))
	except ValueError as e:
		# Missing API key or bad config
		print(f'\nConfiguration error: {e}')
		sys.exit(1)
	except KeyboardInterrupt:
		print('\n\nStopped by user.')
		output_abs = os.path.abspath(args.output)
		if Path(args.output).exists():
			print(f'Partial data saved to: {output_abs}')
		sys.exit(0)
	except Exception as e:
		print(f'\nUnexpected error: {e}')
		raise

	# Final summary
	bar = '=' * 60
	print(f'\n{bar}')
	print('  Scraping complete!')
	print(bar)
	print(f'  Reviews saved : {total:,}')
	print(f'  File          : {os.path.abspath(args.output)}')
	print(f'{bar}\n')


if __name__ == '__main__':
	main()
