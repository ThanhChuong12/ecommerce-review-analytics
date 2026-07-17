"""
crawl.py — Multi-site review crawler for Tiki, Lazada, TGDD, Shopee.

USAGE:
  1. Add target URLs to URLS_TO_CRAWL below
  2. Configure MAX_REVIEWS, OUTPUT_DIR if needed
  3. Run: python crawl.py
"""

# ============================================================
#  LIST OF URLS TO CRAWL
# ============================================================
URLS_TO_CRAWL: list[str] = [
    # --- SHOPEE ---
    #"https://shopee.vn/Senbenbao-Tai-Nghe-Bluetooth-5.3-Kh%C3%B4ng-D%C3%A2y-TWS-Pro4-K%C3%A8m-H%E1%BB%99p-S%E1%BA%A1c-%C4%90i%E1%BB%87n-Tho%E1%BA%A1i-i.888370483.25273889730",
    #"https://shopee.vn/Loa-bluetooth-K12-Kh%C3%B4ng-D%C3%A2y-mini-K%C3%A8m-2-Micro-Thi%E1%BA%BFt-K%E1%BA%BF-Nh%E1%BB%8F-G%E1%BB%8Dn-Ti%E1%BB%87n-D%E1%BB%A5ng-OLIVO-i.664574434.28512742766",
    #"https://shopee.vn/Tai-Nghe-Bluetooth-5.3-Baseus-WM01-TWS-Ch%E1%BB%91ng-%E1%BB%92n-i.131195741.6938221363",
    #"https://shopee.vn/Tai-Nghe-A.K.G-C%C3%B3-D%C3%A2y-Jack-Type-C-3.5mm-H%E1%BB%97-Tr%E1%BB%A3-Android-PC-Laptop-%E2%80%93-M%C3%A0u-%C4%90en-i.533371118.25720935353",
    # --- TIKI ---
    #"https://tiki.vn/dien-thoai-samsung-galaxy-a36-5g-8gb-256gb-p277596856.html",
    #"https://tiki.vn/dien-thoai-xiaomi-redmi-15-8gb-128gb-hang-chinh-hang-p278796276.html",
    #"https://tiki.vn/son-duong-moi-hieu-chinh-ung-hong-tu-nhien-lipice-sheer-color-p33597680.html",
    #"https://tiki.vn/dau-goi-selsun-chong-gau-sach-gau-het-ngua-da-dau-selsun-anti-dandruff-shampoo-50ml-p20541866.html",
    #"https://tiki.vn/sach-nguoi-giau-co-nhat-thanh-babylon-tai-ban-2020-p57325187.html?spid=57325188",
    #"https://tiki.vn/thung-48-hop-sua-nestle-milo-nuoc-180ml-hop-p10240037.html",
    #"https://tiki.vn/thung-sua-dau-nanh-fami-nguyen-chat-200ml-x-36-hop-p12629696.html",

   
    # --- TGDD ---
    #"https://www.thegioididong.com/dtdd/samsung-galaxy-a06-5g-6gb-128gb",
    #"https://www.thegioididong.com/dtdd/iphone-16-pro",
    #"https://www.thegioididong.com/dtdd/iphone-15-pro-max-1tb",
    #"https://www.thegioididong.com/sac-dtdd/pin-sac-du-phong-10000mah-type-c-15w-ava-ds608a?utm_flashsale=1",

    # --- LAZADA ---
    "https://www.lazada.vn/products/pdp-i150498381-s158167954.html",
    "https://www.lazada.vn/products/pdp-i246452966-s316699339.html",
    "https://www.lazada.vn/products/pdp-i2756708-s3347924.html",
    "https://www.lazada.vn/products/trung-nguyen-legend-ca-phe-rang-xay-sang-tao-1-bich-340gr-i353468040-s578424935.html",
    "https://www.lazada.vn/products/pdp-i249064037-s327413856.html",
    "https://www.lazada.vn/products/pdp-i1538401873-s6471204336.html",
    "https://www.lazada.vn/products/pdp-i2763102767-s13733926756.html",
    "https://www.lazada.vn/products/pdp-i1597967647-s6853265058.html",
    "https://www.lazada.vn/products/pdp-i1465875113-s6080623320.html",

]

# ============================================================
#  CONFIGURATION
# ============================================================
MAX_REVIEWS: int     = 0   # 0 fetch all reviews per URL
OUTPUT_DIR: str      = "data"   # Output folder path
LAZADA_HEADLESS: bool = False    # Run visible to solve captchas manually
SHOPEE_FILTER: str   = "max"     # all comment media max
# ============================================================

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

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=False))

from scraper.dispatcher import scrape as _dispatch


# ----------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------

def _site_label(url: str) -> str:
    """Get short site name for naming files."""
    url = url.lower()
    if "tiki.vn" in url:
        return "tiki"
    if "thegioididong.com" in url:
        return "tgdd"
    if "lazada.vn" in url:
        return "lazada"
    if "shopee.vn" in url:
        return "shopee"
    return "unknown"


def _safe_filename(url: str) -> str:
    """Generate safe CSV filename from URL."""
    import re
    label = _site_label(url)
    # Remove protocol
    slug = url.split("//", 1)[-1]
    slug = re.sub(r"[^a-zA-Z0-9_\-]", "_", slug)
    slug = re.sub(r"_+", "_", slug).strip("_")
    slug = slug[:60]                        # Limit length
    ts   = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{label}_{slug}_{ts}.csv"


def _row_key(row: dict) -> str:
    """Generate dedup key from review content."""
    import hashlib
    raw = "|".join([
        str(row.get("text", ""))[:200],
        str(row.get("rating", "")),
        str(row.get("date", "")),
        str(row.get("product_url", "")),
    ])
    return hashlib.md5(raw.encode("utf-8", errors="replace")).hexdigest()


def _load_existing_keys(path: Path) -> set[str]:
    """Load existing dedup keys from all_reviews.csv."""
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
    """
    Append reviews from csv_files to out_path.
    Duplicates are skipped. Returns (added, skipped).
    """
    # Load existing keys from master file
    existing_keys = _load_existing_keys(out_path)
    old_count = len(existing_keys)

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
                    # Write header for new files only
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
    print("  E-commerce Review Crawler — Tiki / TGDD / Lazada")
    print(bar)
    print(f"  URLs      : {len(URLS_TO_CRAWL)}")
    print(f"  Max/URL   : {MAX_REVIEWS:,} reviews")
    print(f"  Output    : {Path(OUTPUT_DIR).resolve()}")
    print(f"  Lazada    : headless={LAZADA_HEADLESS}")
    print(bar)
    print("  Press Ctrl+C to stop crawling at any time.\n")


# ----------------------------------------------------------------
# Core async runner
# ----------------------------------------------------------------

async def _crawl_one(
    url: str,
    output_path: Path,
    idx: int,
    total: int,
) -> int:
    """Scrape a single URL. Returns saved review count."""
    bar = "-" * 65
    print(f"\n{bar}")
    print(f"  [{idx}/{total}] {_site_label(url).upper()} → {url[:80]}")
    print(f"  Saving to: {output_path.name}")
    print(bar)

    # Pass headless parameter to dispatcher
    total_saved = await _dispatch(
        url=url,
        output_path=str(output_path),
        fmt="csv",
        max_reviews=MAX_REVIEWS,
        llm_provider="auto",
        headless=LAZADA_HEADLESS,
        filter_mode=SHOPEE_FILTER,
    )
    return total_saved


async def main() -> None:
    if not URLS_TO_CRAWL:
        print("\n[!] URLS_TO_CRAWL is empty!")
        print("    Please add URLs to crawl.py and try again.\n")
        sys.exit(1)

    _print_banner()

    out_dir = (_THIS_DIR / OUTPUT_DIR)
    out_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []   # {"url", "file", "count", "status"}
    csv_files: list[Path] = []

    for idx, url in enumerate(URLS_TO_CRAWL, start=1):
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

    # -- Append to master file
    MASTER_FILE = "all_reviews.csv"
    merged_path = out_dir / MASTER_FILE
    added, skipped = _append_to_master(csv_files, merged_path)
    print(f"\n  [all_reviews.csv] +{added:,} new, skipped {skipped:,} duplicates → {merged_path}")

    # -- Print summary
    bar = "=" * 65
    print(f"\n{bar}")
    print("  SUMMARY RESULTS")
    print(bar)
    print(f"  {'URL':<55} {'Reviews':>8}  Status")
    print(f"  {'-'*54} {'-'*8}  {'-'*10}")
    for r in results:
        url_short = r["url"][:54]
        print(f"  {url_short:<55} {r['count']:>8,}  {r['status']}")
    total_reviews = sum(r["count"] for r in results)
    print(f"  {'':55} {'─'*8}")
    print(f"  {'TOTAL':<55} {total_reviews:>8,}")
    print(bar)
    print(f"  All CSV files saved to : {out_dir.resolve()}")
    print(f"  Master combined file   : {MASTER_FILE}")
    print(f"{bar}\n")


if __name__ == "__main__":
    asyncio.run(main())
