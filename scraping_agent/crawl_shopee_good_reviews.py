"""
crawl_shopee_good_reviews.py — Tự động cào đánh giá (4 sao, 5 sao) từ Shopee

CÁCH DÙNG:
  1. Điền danh sách 10 URL Shopee vào URLS_TO_CRAWL bên dưới
  2. Chạy: python crawl_shopee_good_reviews.py

Kết quả: mỗi URL được lưu vào 1 file CSV riêng trong thư mục output (data).
"""

import asyncio
import os
import sys
import csv
from datetime import datetime
from pathlib import Path

# Fix encoding Windows
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")
if hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8")

# Đảm bảo import được các module trong scraping_agent/
_THIS_DIR = Path(__file__).parent
sys.path.insert(0, str(_THIS_DIR))

# File session Shopee — sẽ bị xóa sau mỗi link để reset fingerprint
_SESSION_FILE = _THIS_DIR / "output" / "agent_sessions" / "state_shopee.vn.json"

from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv(usecwd=False))

# --- MONKEY PATCH ĐỂ CHỈ CÀO 4*, 5* ---
# Can thiệp vào module shopee_fast trước khi chạy scraper
from scraper.direct import shopee_fast
shopee_fast.STAR_TYPES = [4, 5]
# ------------------------------------------

from scraper.dispatcher import scrape as _dispatch

# ============================================================
#  ĐẶT DANH SÁCH 10 URL SHOPEE VÀO ĐÂY
# ============================================================
URLS_TO_CRAWL: list[str] = [
    # Nhập 10 links ở đây
    "https://shopee.vn/Tai-Nghe-A.K.G-C%C3%B3-D%C3%A2y-Jack-Type-C-3.5mm-H%E1%BB%97-Tr%E1%BB%A3-Android-PC-Laptop-%E2%80%93-M%C3%A0u-%C4%90en-i.533371118.25720935353",
    "https://shopee.vn/%C3%81o-thun-tr%C6%A1n-Kirk.L-A-N-D-Unisex-Oversize-Nam-N%E1%BB%AF-%C3%81o-ph%C3%B4ng-c%E1%BB%95-tr%C3%B2n-M%C3%A0u--i.63569693.2237785253",
    "https://shopee.vn/%C3%81o-Sweater-Frozen-Night-Life-N%E1%BB%89-Ch%C3%A2n-Cua-l%C3%B3t-l%C3%B4ng-Cotton-100-Unisex-Local-Brand-i.564687320.27961145217",
    "https://shopee.vn/Tai-nghe-ng%E1%BB%A7-X55-Tai-nghe-l%C3%A0m-vi%E1%BB%87c-mini-kh%C3%B4ng-d%C3%A2y-TWS-Tai-nghe-v%E1%BB%9Bi-Microphone-i.1574448654.28488599214",
    "https://shopee.vn/Tai-Nghe-C%C3%B3-D%C3%A2y-S2000-Gaming-Super-Bass-Ch%E1%BB%91ng-%E1%BB%92n-Hi%E1%BB%87u-Qu%E1%BA%A3-C%C3%B3-Mic-%C4%90%C3%A0m-Tho%E1%BA%A1i-i.145181001.8986374131",
    #"https://shopee.vn/(T%E1%BA%B7ng-14-Charm)-D%C3%A9p-S%E1%BB%A5c-Xchill-%C4%90%E1%BA%BF-Th%E1%BA%A5p-Form-Basic-Unisex-Mang-%C3%8Am-Ch%C3%A2n-i.1430085998.27470964637",
    # "https://shopee.vn/%C4%90%E1%BB%93ng-h%E1%BB%93-nam-n%E1%BB%AF-Unisex-Led-ki%E1%BB%83u-d%C3%A1ng-phi-h%C3%A0nh-gia-d%C3%A2y-cao-su-%C3%AAm-tay-th%E1%BB%9Di-trang-c%C3%A1-t%C3%ADnh-ROSE29-i.1124272918.25906053400",
    # "https://shopee.vn/%C4%90%E1%BB%93ng-h%E1%BB%93-%C4%91i%E1%BB%87n-t%E1%BB%AD-WR-F94WA-9DG-ch%E1%BB%91ng-n%C6%B0%E1%BB%9Bc-b%C6%A1i-l%E1%BB%99i-%C4%91i-m%C6%B0a-tho%E1%BA%A3i-m%C3%A1i-%C4%90%E1%BB%93ng-h%E1%BB%93-F94-d%C3%A2y-nh%E1%BB%B1a-huy%E1%BB%81n-tho%E1%BA%A1i.-i.135375573.2915547104",
    # "https://shopee.vn/t%C3%BAi-%C4%91eo-ch%C3%A9o-t%C3%BAi-x%C3%A1ch-n%E1%BB%AF-c%E1%BA%A7m-tay-jac-da-l%C3%AC-%E1%BA%A3nh-th%E1%BA%ADt-%C4%91%E1%BB%A7-m%C3%A0u-b%E1%BB%81n-%C4%91%E1%BA%B9p-t%C3%BAi-v%E1%BB%ABa-%C4%91i%E1%BB%87n-tho%E1%BA%A1i-th%E1%BB%9Di-trang-d%E1%BB%85-d%C3%B9ng-i.989635088.28501668253",
    # "https://shopee.vn/T%C3%BAi-x%C3%A1ch-n%E1%BB%AF-mini-da-nhung-H%C3%A0n-Qu%E1%BB%91c-d%C3%A1ng-b%E1%BA%A7u-k%C3%A8m-g%E1%BA%A5u-b%C3%B4ng-treo-n%C6%A1-xinh-%E2%80%93-%C4%91i-h%E1%BB%8Dc-%C4%91i-l%C3%A0m-B%C3%A1o-Store-TXN143-i.1488368324.24496799253",
    # "https://shopee.vn/-Gi%C3%A1-m%E1%BB%9F-b%C3%A1n-Balo-da-PU-%C4%91i-h%E1%BB%8Dc-%C4%91i-ch%C6%A1i-ki%E1%BB%83u-d%C3%A1ng-Basic-cho-nam-n%E1%BB%AF-41x30x13cm-balo-BL48-i.120761904.29956470951",
    # "https://shopee.vn/%C4%90%E1%BB%93ng-h%E1%BB%93-nam-ch%C3%ADnh-h%C3%A3ng-PABLO-RAEZ-d%C3%A2y-da-cao-c%E1%BA%A5p-c%C3%B3-l%E1%BB%8Bch-ng%C3%A0y-d%E1%BA%A1-quang-cao-c%E1%BA%A5p-U850-CARIENT-i.535470558.22864673701",
    
    
    #"https://shopee.vn/%E1%BB%90p-L%C6%B0ng-iPhone-TPU-Silicon-M%E1%BB%81m-4-G%C3%B3c-Cao-C%E1%BA%A5p-Ch%E1%BB%91ng-S%E1%BB%91c-B%E1%BA%A3o-V%E1%BB%87-Camera-iP-6-7-8-Plus-X-XS-11-12-13-14-15-16-17-Pro-Max-i.89827191.23244410073",
    #"https://shopee.vn/%E1%BB%90p-L%C6%B0ng-iPhone-%C4%90%E1%BB%B1ng-%E1%BA%A2nh-Th%E1%BA%BB-Cao-C%E1%BA%A5p-Ch%E1%BB%91ng-S%E1%BB%91c-B%E1%BA%A3o-V%E1%BB%87-Camera-7-8-X-XS-11-12-13-14-15-16-17-Plus-Pro-Max-AWiFi-G5-6-i.7669738.16359354297",
    # "https://shopee.vn/%E1%BB%90p-l%C6%B0ng-iphone-TPU-Silicon-B%E1%BA%A3o-V%E1%BB%87-B%E1%BB%91n-G%C3%B3c-Trong-Si%C3%AAu-Ch%E1%BB%91ng-S%E1%BB%91c-6plus-7plus-8plus-x-xs-11-12-13-14-15-16-17-pro-max-U4-13-i.7669738.18381625974",
    # "https://shopee.vn/%E1%BB%90p-l%C6%B0ng-iphone-vi%E1%BB%81n-camera-kim-lo%E1%BA%A1i-6-6splus-7-7plus-8-8plus-x-xs-11-12-13-14-15-16-17-pro-max-plus-promax-Awifi-R5-6-i.7669738.23248488138",
    # "https://shopee.vn/%C3%81o-Thun-Tr%C6%A1n-Cotton-250gsm-GODMOTHER-Premium-Full-Color-C%E1%BB%95-Tr%C3%B2n-Nam-N%E1%BB%AF-i.703090265.19787998337",
    # "https://shopee.vn/%C3%81o-Thun-Local-Brand-VIBESTU-Only-Members-%C3%81o-Thun-Boxy-Form-R%E1%BB%99ng-Unisex-250Gsm-Cotton-i.1173412191.29614790499",
    # "https://shopee.vn/%C3%81o-thun-cotton-nam-n%E1%BB%AF-JULIDO-ch%E1%BA%A5t-li%E1%BB%87u-v%E1%BA%A3i-thu%E1%BA%A7n-coton-tho%C3%A1ng-m%C3%A1t-i.33435563.25572468136",
    # "https://shopee.vn/%C3%81O-THUN-BOXY-DAMIESE-IN-PH%E1%BB%92NG-COTTON-260GSM-i.46532250.25393996064",
    # "https://shopee.vn/%C3%81o-thun-nam-th%E1%BB%83-thao-t%E1%BA%ADp-Gym-Tennis-Pickleball-Coolmate-Basics-tho%E1%BA%A3i-m%C3%A1i-th%E1%BA%A5m-h%C3%BAt-nhanh-kh%C3%B4-i.24710134.10280321597",

    # "https://shopee.vn/-HOT-COMPO-10-%E1%BB%90c-Salaya-5li-6li1-6li15-Inox-304-M%E1%BA%ABu-Th%C3%A1i-K%C3%A8m-L%C3%B4ng-%C4%90%E1%BB%81n-Kitaco-i.114067592.15055331754",
    # "https://shopee.vn/Combo-L%C3%B4ng-%C4%91%E1%BB%81n-%C4%91%E1%BB%8F-nh%C3%B4m-KITACO-ch%C3%ADnh-h%C3%A3ng-th%C3%B4ng-d%E1%BB%A5ng-c%E1%BB%B1c-hot-xu-h%C6%B0%E1%BB%9Bng-hi%E1%BB%87n-nay-(-ko-bao-g%E1%BB%93m-%E1%BB%91c-)-i.778873826.20214439417",
    # "https://shopee.vn/%C4%90%C3%A8n-Led-G%E1%BA%AFn-Van-Xe-1-c%C3%A1i-i.19609604.22048136056",
    # "https://shopee.vn/%C4%90%C3%A8n-led-g%E1%BA%AFn-b%C3%A1nh-xe-%C4%91%E1%BA%A1p-ch%E1%BB%91ng-th%E1%BA%A5m-n%C6%B0%E1%BB%9Bc-3-ch%E1%BA%BF-%C4%91%E1%BB%99-s%C3%A1ng-i.3075268.5535318985",
    # "https://shopee.vn/C%E1%BA%A2NG-SAU-SIRUS-KI%E1%BB%82U-SPARK-TH%C3%81I-i.321951464.7659738719",
    # "https://shopee.vn/32650-dung-l%C6%B0%E1%BB%A3ng-6000-t%E1%BA%B7ng-k%C3%A8m-%E1%BB%91c-v%C3%ADt-i.1543750459.42451490233",
    #"https://shopee.vn/Tai-nghe-bluetooth-KY9-KY8-Ultrapods-Pro-c%E1%BA%A3m-%E1%BB%A9ng-bass-hay-si%C3%AAu-nh%E1%BB%8F-g%E1%BB%8Dn-c%C3%B3-mic-%C4%91%C3%A0m-tho%E1%BA%A1i-nghe-nh%E1%BA%A1c-c%E1%BB%B1c-hay-i.1385065291.26470082188",
    #"https://shopee.vn/%E1%BB%90c-inox-304-salaya-m%E1%BA%ABu-th%C3%A1i-5ly-6ly-l%E1%BA%AFp-d%C3%A0n-%C3%A1o-c%C3%A1c-lo%E1%BA%A1i-xe-i.831605251.26430579253",
]

# ============================================================
#  CẤU HÌNH
# ============================================================
MAX_REVIEWS: int     = 0         # 0 = fetch all available reviews per URL
OUTPUT_DIR: str      = "data"    # Thư mục lưu CSV
SHOPEE_FILTER: str   = "max"     # all | comment | media | max
# ============================================================


# ----------------------------------------------------------------
# Helpers (Tái sử dụng từ crawl.py)
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
    return f"{label}_{slug}_good_reviews_{ts}.csv"

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
    print("  Shopee Good Reviews Crawler (4*, 5*)")
    print(bar)
    print(f"  URLs      : {len(URLS_TO_CRAWL)}")
    print(f"  Output    : {Path(OUTPUT_DIR).resolve()}")
    print(bar)
    print("  Nhấn Ctrl+C bất kỳ lúc nào để dừng.\n")

# ----------------------------------------------------------------
# Core async runner
# ----------------------------------------------------------------

async def _crawl_one(url: str, output_path: Path, idx: int, total: int) -> int:
    import random
    bar = "-" * 65
    print(f"\n{bar}")
    print(f"  [{idx}/{total}] Đang cào -> {url[:80]}")
    print(f"  Đang lưu vào: {output_path.name}")
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

    # Xóa session sau mỗi link để Shopee không nhận ra fingerprint cũ
    if _SESSION_FILE.exists():
        _SESSION_FILE.unlink()
        print(f"  [Session] Đã xóa session cũ — link tiếp theo sẽ dùng session mới.")

    # Nghỉ ngẫu nhiên 15-30 giây trước link tiếp theo để tránh rate-limit
    if idx < len([u for u in URLS_TO_CRAWL if u.strip() and not u.strip().startswith('#')]):
        wait = random.uniform(15, 30)
        print(f"  [Delay] Nghỉ {wait:.0f}s trước link tiếp theo...")
        await asyncio.sleep(wait)

    return total_saved

async def main() -> None:
    if not URLS_TO_CRAWL:
        print("\n[!] URLS_TO_CRAWL đang trống!\n")
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
            print("\n[!] Dừng theo yêu cầu người dùng.")
            results.append({"url": url, "file": out_file.name, "count": 0, "status": "STOPPED"})
            break
        except Exception as exc:
            print(f"\n[!] Lỗi khi cào {url}: {exc}")
            results.append({"url": url, "file": out_file.name, "count": 0, "status": f"ERROR: {exc}"})

    # Append
    MASTER_FILE = "all_good_reviews.csv"
    merged_path = out_dir / MASTER_FILE
    added, skipped = _append_to_master(csv_files, merged_path)
    print(f"\n  [{MASTER_FILE}] +{added:,} mới, bỏ qua {skipped:,} trùng -> {merged_path}")

    # Kết quả
    bar = "=" * 65
    print(f"\n{bar}")
    print("  KẾT QUẢ TỔNG KẾT")
    print(bar)
    for r in results:
        url_short = r["url"][:54]
        print(f"  {url_short:<55} {r['count']:>8,}  {r['status']}")
    total_reviews = sum(r["count"] for r in results)
    print(f"  {'TỔNG':<55} {total_reviews:>8,}")
    print(bar)

if __name__ == "__main__":
    asyncio.run(main())
