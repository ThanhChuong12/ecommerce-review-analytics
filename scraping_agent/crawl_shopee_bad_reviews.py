"""
crawl_shopee_bad_reviews.py — Crawl 1, 2, and 3 star reviews from Shopee.

USAGE:
  1. Add Shopee product URLs to URLS_TO_CRAWL below
  2. Run: python crawl_shopee_bad_reviews.py
"""

import asyncio
import os
import sys
import csv
from datetime import datetime
from pathlib import Path

# Configure Windows terminal encoding
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Ensure scraping_agent is in Python path
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

# Shopee session file path
_SESSION_FILE = _THIS_DIR / "output" / "agent_sessions" / "state_shopee.vn.json"

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=False))

# --- Monkeypatch to only crawl 1*, 2*, 3* reviews ---
from scraper.direct import shopee_fast
shopee_fast.STAR_TYPES = [1, 2, 3]
# ----------------------------------------------------

from scraper.dispatcher import scrape as _dispatch

# ============================================================
#  LIST OF URLS TO CRAWL
# ============================================================
URLS_TO_CRAWL: list[str] = [
    # Add URLs here
    "https://shopee.vn/%E1%BB%9Cc-inox-304-salaya-m%E1%BA%ABu-th%C3%A1i-5ly-6ly-l%E1%BA%AFp-d%C3%A0n-%C3%A1o-c%C3%A1c-lo%E1%BA%A1i-xe-i.831605251.26430579253",
]

# ============================================================
#  CONFIGURATION
# ============================================================
MAX_REVIEWS: int     = 0         # 0 = fetch all available reviews per URL
OUTPUT_DIR: str      = "data"    # Output folder path
SHOPEE_FILTER: str   = "max"     # all | comment | media | max
# ============================================================


# ----------------------------------------------------------------
# Helpers (Reused from crawl.py)
# ----------------------------------------------------------------

def _site_label(url: str) -> str:
    return "shopee"

def _safe_filename(url: str) -> str:
    import re
    label = _site_label(url)
    slug = url.split("//", 1)[-1]
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    slug = slug[:60]
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{label}_{slug}_bad_reviews_{ts}.csv"

def _row_key(row: dict) -> str:
    import hashlib
    raw = "|".join([
        str(row.get("text", ""))[:200],
        str(row.get("rating", "")),
        str(row.get("date", "")),
        str(row.get("product_url", "")),
    ])
    return hashlib.md5(raw.encode("utf-8", errors="replace")).hexdigest()

def _load_existing_keys(path: Path) -> set[str]:
    seen: set[str] = set()
    if not path.exists():
        return seen
    try:
        with open(path, newline="", encoding="utf-8-sig") as f:
            for row in csv.DictReader(f):
                seen.add(_row_key(row))
    except Exception:
        pass
    return seen

def _append_to_master(csv_files: list[Path], out_path: Path) -> tuple[int, int]:
    existing_keys = _load_existing_keys(out_path)
    file_exists = out_path.exists()
    added = 0
    skipped = 0
    fieldnames = None

    with open(out_path, "a", newline="", encoding="utf-8-sig") as fout:
        writer = None
        for fp in csv_files:
            if not fp.exists():
                continue
            with open(fp, newline="", encoding="utf-8-sig") as fin:
                reader = csv.DictReader(fin)
                if fieldnames is None:
                    fieldnames = reader.fieldnames
                if writer is None:
                    writer = csv.DictWriter(fout, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
                    if not file_exists:
                        writer.writeheader()
                for row in reader:
                    key = _row_key(row)
                    if key in existing_keys:
                        skipped += 1
                    else:
                        existing_keys.add(key)
                        writer.writerow(row)
                        added += 1

    return added, skipped

def _print_banner() -> None:
    bar = "=" * 65
    print(f"\n{bar}")
    print("  Shopee Bad Reviews Crawler (1*, 2*, 3*)")
    print(bar)
    print(f"  URLs      : {len(URLS_TO_CRAWL)}")
    print(f"  Output    : {Path(OUTPUT_DIR).resolve()}")
    print(bar)
    print("  Press Ctrl+C to stop crawling at any time.\n")

# ----------------------------------------------------------------
# Core async runner
# ----------------------------------------------------------------

async def _crawl_one(url: str, output_path: Path, idx: int, total: int) -> int:
    import random
    bar = "-" * 65
    print(f"\n{bar}")
    print(f"  [{idx}/{total}] Crawling -> {url[:80]}")
    print(f"  Saving to: {output_path.name}")
    print(bar)

    total_saved = await _dispatch(
        url=url,
        output_path=str(output_path),
        fmt="csv",
        max_reviews=MAX_REVIEWS,
        llm_provider="auto",
        headless=False,
        filter_mode=SHOPEE_FILTER,
    )

    # Delay between 15 and 30 seconds to prevent rate limiting
    if idx < len([u for u in URLS_TO_CRAWL if u.strip() and not u.strip().startswith('#')]):
        wait = random.uniform(15, 30)
        print(f"  [Delay] Waiting {wait:.0f}s before next link...")
        await asyncio.sleep(wait)

    return total_saved

async def main() -> None:
    if not URLS_TO_CRAWL:
        print("\n[!] URLS_TO_CRAWL is empty!\n")
        sys.exit(1)

    _print_banner()

    out_dir = (_THIS_DIR / OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []
    csv_files: list[Path] = []

    for idx, url in enumerate(URLS_TO_CRAWL, start=1):
        if url.strip().startswith("#") or not url.strip():
            continue
        out_file = out_dir / _safe_filename(url)
        try:
            count = await _crawl_one(url, out_file, idx, len(URLS_TO_CRAWL))
            results.append({"url": url, "file": out_file.name, "count": count, "status": "OK"})
            csv_files.append(out_file)
        except (KeyboardInterrupt, asyncio.CancelledError):
            print("\n[!] Interrupted by user.")
            results.append({"url": url, "file": out_file.name, "count": 0, "status": "STOPPED"})
            break
        except Exception as exc:
            print(f"\n[!] Error crawling {url}: {exc}")
            results.append({"url": url, "file": out_file.name, "count": 0, "status": f"ERROR: {exc}"})

    # Append
    MASTER_FILE = "all_bad_reviews.csv"
    merged_path = out_dir / MASTER_FILE
    added, skipped = _append_to_master(csv_files, merged_path)
    print(f"\n  [{MASTER_FILE}] +{added:,} new, skipped {skipped:,} duplicates -> {merged_path}")

    # Summary
    bar = "=" * 65
    print(f"\n{bar}")
    print("  SUMMARY RESULTS")
    print(bar)
    for r in results:
        url_short = r["url"][:54]
        print(f"  {url_short:<55} {r['count']:>8,}  {r['status']}")
    total_reviews = sum(r["count"] for r in results)
    print(f"  {'TOTAL':<55} {total_reviews:>8,}")
    print(bar)

if __name__ == "__main__":
    asyncio.run(main())
